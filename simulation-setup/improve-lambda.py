"""Improve the replica exchange lambda schedule from the overlap of the sampled states."""

import argparse
import gzip
import pickle
from pathlib import Path

import matplotlib
import numpy as np
from scipy.special import erfc, erfcinv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SIMULATION_PATH = ROOT / "simulations"
SUBDIR = "waterS-r1"  # repeat to read the windows from, protein-r1 for the protein leg
REF_TARGET = "bace_CAT-13a"  # charged ligand whose schedule is taken as the reference
REF_MDP = SIMULATION_PATH / REF_TARGET / SUBDIR / "production.mdp"
# the vdw leg is evaluated first, which fixes the random pairings of both legs
LEGS = ("vdw", "coul")
MDP_KEYS = ("coul-lambdas", "vdw-lambdas", "restraint-lambdas")
EQUIL_FRAC = 0.2  # discarded initial fraction of each window
MAX_FRAMES = 5000  # subsampled frames per state for the estimate
PAIR_REPS = 4  # averaged random pairings
SEED = 0
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]
OUT = Path(__file__).resolve().parent / "improve-lambda-out" / SUBDIR
RND = 5  # rounding for state-tuple keys
ATOL = 1e-3  # value tolerance when matching a target's grid to the reference


def parse_ref_mdp(path: Path) -> dict:
    """
    Read the reference schedule and return each leg's sorted lambda values.
    :param path: Path to the mdp file holding the reference schedule.
    :return: Sorted lambda values of the charge and the vdw leg, and the restraint lead.
    """
    vec = {}
    for line in path.read_text().splitlines():
        for key in MDP_KEYS:
            if line.strip().startswith(key):
                vec[key] = line.split("=", 1)[1].split()
    if "coul-lambdas" not in vec:
        raise ValueError(f"{path} has no charge leg, the reference must be charged")
    coul = [float(x) for x in vec["coul-lambdas"]]
    vdw = [float(x) for x in vec["vdw-lambdas"]]
    res = vec.get("restraint-lambdas", ["0.0"] * len(vdw))
    restraint = [float(x) for x in res]
    cmax = max(coul)
    states = list(zip(coul, vdw, restraint))
    return {
        # charge leg: restraint 0 and vdw 0; vdw leg: restraint 0 and coul at its maximum
        "coul_grid": sorted(c for c, v, r in states if r == 0.0 and v == 0.0),
        "vdw_grid": sorted(v for c, v, r in states if r == 0.0 and c == cmax),
        # restraint lead: switched before the alchemical legs, kept verbatim in the output
        "restraint_lead": [s for s, r in zip(res, restraint) if r > 0.0],
    }


def load_target(tdir: Path) -> tuple:
    """
    Load the sampled reduced potentials of all windows of one target. Handles both the
    charged index (coul, vdw, restraint) and the neutral one (vdw, restraint).
    :param tdir: Directory of the target holding the lambda windows.
    :return: Samples per state and, per leg, its sorted states with their lambda values.
    """
    samples, states, names = {}, [], None
    for d in sorted(tdir.glob("lambda-*")):
        pkl = d / "production.pkl.gz"
        if not pkl.exists():
            continue
        with gzip.open(pkl, "rb") as f:
            df = pickle.load(f)
        names = [n for n in df.index.names if n != "time"]
        st = df.index.droplevel("time").unique()
        if len(st) != 1:
            raise ValueError(f"{d.name}: expected 1 state, got {len(st)}")
        state = tuple(round(float(x), RND) for x in np.atleast_1d(st[0]))
        # drop equilibration then subsample frames
        df = df.iloc[int(len(df) * EQUIL_FRAC) :]
        if len(df) > MAX_FRAMES:
            df = df.iloc[np.linspace(0, len(df) - 1, MAX_FRAMES).astype(int)]
        df.columns = [
            tuple(round(float(c), RND) for c in np.atleast_1d(col))
            for col in df.columns
        ]
        samples[state] = df
        states.append(state)
    if names is None:
        raise ValueError("no windows found")
    # locate each lambda component from the index names
    idx = {n: i for i, n in enumerate(names)}
    iv, ir, ic = idx["vdw-lambda"], idx.get("restraint-lambda"), idx.get("coul-lambda")
    free = states if ir is None else [s for s in states if s[ir] == 0.0]
    if ic is None:  # neutral: no charge leg
        vdw_states, coul_states = sorted(free, key=lambda s: s[iv]), []
    else:  # charged: vdw leg at max coul, charge leg at vdw==0
        cmax = max(s[ic] for s in free)
        vdw_states = sorted((s for s in free if s[ic] == cmax), key=lambda s: s[iv])
        coul_states = sorted((s for s in free if s[iv] == 0.0), key=lambda s: s[ic])
    return samples, {
        "vdw": (vdw_states, [s[iv] for s in vdw_states]),
        "coul": (coul_states, [s[ic] for s in coul_states] if ic is not None else []),
    }


def pair_acceptance(samples: dict, a: tuple, b: tuple, rng) -> float:
    """
    Predicted Metropolis exchange acceptance between two states from paired samples.
    :param samples: Reduced potentials of every state, as returned by load_target.
    :param a: State to exchange from.
    :param b: State to exchange to.
    :param rng: Random generator used to pair the samples of both states.
    :return: Mean acceptance over PAIR_REPS random pairings.
    """
    da, db = samples[a], samples[b]
    wf = (da[b] - da[a]).to_numpy()  # forward work from a-samples
    wr = (db[a] - db[b]).to_numpy()  # reverse work from b-samples
    m = min(len(wf), len(wr))
    accs = []
    for _ in range(PAIR_REPS):
        delta = wf[rng.permutation(len(wf))[:m]] + wr[rng.permutation(len(wr))[:m]]
        accs.append(np.where(delta <= 0, 1.0, np.exp(-np.clip(delta, 0, 700))).mean())
    return float(np.mean(accs))


def leg_matrix(samples: dict, states: list, rng) -> np.ndarray:
    """
    Full pairwise acceptance matrix over a leg's states, indexed by sorted position.
    :param samples: Reduced potentials of every state, as returned by load_target.
    :param states: Sorted states of the leg.
    :param rng: Random generator used to pair the samples of two states.
    :return: Symmetric matrix of exchange acceptances.
    """
    n = len(states)
    A = np.full((n, n), np.nan)
    for i in range(n):
        A[i, i] = 1.0
        for j in range(i + 1, n):
            A[i, j] = A[j, i] = pair_acceptance(samples, states[i], states[j], rng)
    return A


def grids_match(vals: list, ref_vals: list) -> bool:
    """
    Check whether a target's leg values match the reference in count and to ATOL.
    :param vals: Lambda values of the leg of one target.
    :param ref_vals: Lambda values of the reference schedule.
    :return: Whether both schedules agree.
    """
    return len(vals) == len(ref_vals) and np.allclose(
        sorted(vals), sorted(ref_vals), atol=ATOL
    )


def redistribute(adj_acc: np.ndarray, grid: list, target: float) -> tuple:
    """
    Place windows at equal Gaussian thermodynamic length to reach the target acceptance
    uniformly. Acceptance a and swap length s satisfy a = erfc(s / (2 * sqrt(2))), so s
    scales with erfcinv(a) and the constant factor cancels in the window placement.
    :param adj_acc: Acceptance of each adjacent pair of the original schedule.
    :param grid: Lambda values of the original schedule.
    :param target: Exchange acceptance to reach between adjacent windows.
    :return: New lambda values and the acceptance predicted between them.
    """
    grid = np.array(grid)
    # cumulative length from the worst-case acceptance of each original adjacent pair
    L = np.concatenate([[0.0], np.cumsum(erfcinv(np.clip(adj_acc, 1e-6, 1 - 1e-9)))])
    nseg = max(1, int(np.ceil(L[-1] / erfcinv(target))))
    lam = np.interp(np.linspace(0.0, L[-1], nseg + 1), L, grid)
    lam[0], lam[-1] = grid[0], grid[-1]
    return lam, float(erfc(L[-1] / nseg))


def compute_all(ref: dict, recompute: bool) -> dict:
    """
    Compute the acceptance matrices of both legs for every target, or load them from cache.
    :param ref: Reference schedule, as returned by parse_ref_mdp.
    :param recompute: Rebuild the cache instead of loading it.
    :return: Acceptance matrix per target, for each leg.
    """
    cache = OUT / "accept_cache.npz"
    if cache.exists() and not recompute:
        d = np.load(cache, allow_pickle=True)
        return {leg: d[leg].item() for leg in LEGS}
    by_leg = {leg: {} for leg in LEGS}
    targets = sorted(
        p.parent.parent.name for p in SIMULATION_PATH.glob(f"*/{SUBDIR}/production.mdp")
    )
    for ti, name in enumerate(targets):
        rng = np.random.default_rng(SEED)
        try:
            samples, target_legs = load_target(SIMULATION_PATH / name / SUBDIR)
            for leg in LEGS:
                leg_states, leg_vals = target_legs[leg]
                # the charge leg only exists for the charged variant
                if leg == "coul" and len(leg_states) <= 1:
                    continue
                if grids_match(leg_vals, ref[f"{leg}_grid"]):
                    by_leg[leg][name] = leg_matrix(samples, leg_states, rng)
                else:
                    print(
                        f"  ! {name}: {leg} grid mismatch "
                        f"({len(leg_states)} states), skipping {leg}"
                    )
        except Exception as e:  # noqa: BLE001
            print(f"  ! {name}: {e}")
        print(
            f"[{ti + 1}/{len(targets)}] {name}: "
            + " ".join(f"{leg}={'y' if name in by_leg[leg] else '-'}" for leg in LEGS)
        )
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(cache, coul=by_leg["coul"], vdw=by_leg["vdw"])
    return by_leg


def summarize_leg(by_target: dict, grid: list) -> dict:
    """
    Aggregate the acceptance matrices of all targets of one leg.
    :param by_target: Acceptance matrix per target.
    :param grid: Lambda values of the leg.
    :return: The grid, the number of targets, their elementwise worst-case matrix and the
        adjacent-pair acceptances with their worst case and median across the targets.
    """
    dist = np.stack([np.diag(m, 1) for m in by_target.values()])
    return {
        "grid": grid,
        "n_targets": len(by_target),
        "worst": np.nanmin(np.stack(list(by_target.values())), axis=0),
        "dist": dist,
        "adj": dist.min(axis=0),
        "median": np.median(dist, axis=0),
    }


def report_distribution(name: str, leg: dict, target: float):
    """
    Print the acceptance percentiles of every pair and flag the pairs below the target.
    :param name: Name of the leg.
    :param leg: Summary of the leg, as returned by summarize_leg.
    :param target: Exchange acceptance to reach between adjacent windows.
    """
    grid, dist = leg["grid"], leg["dist"]
    print(f"\n{name} adjacent-pair acceptance across {dist.shape[0]} targets:")
    print(f" pair  lam_lo->lam_hi    min   p10   p50   max   n<{target:.2f}")
    for k, c in enumerate(dist.T):
        print(
            f" {k:2d}   {grid[k]:.3f}->{grid[k + 1]:.3f}   "
            f"{c.min():.2f}  {np.percentile(c, 10):.2f}  {np.median(c):.2f}  {c.max():.2f}"
            f"   {(c < target).sum():3d}"
        )
    low_pairs = (dist < target).any(axis=0).sum()
    low_targets = (dist < target).any(axis=1).sum()
    print(
        f" -> {low_pairs}/{dist.shape[1]} pairs dip below {target} for some ligand; "
        f"{low_targets}/{dist.shape[0]} ligands dip below on some pair"
    )


def build_mdp(ref: dict, sel_coul: np.ndarray, sel_vdw: np.ndarray) -> tuple:
    """
    Assemble the mdp lambda vectors of the neutral and the charged variant, prepending the
    restraint lead of the reference and keeping its restraint line only if it exists.
    :param ref: Reference schedule, as returned by parse_ref_mdp.
    :param sel_coul: Lambda values of the charge leg.
    :param sel_vdw: Lambda values of the vdw leg.
    :return: Lambda vectors of the neutral and the charged variant.
    """
    lead = ref["restraint_lead"]
    ln, z = len(lead), "0.00000"
    vtok = [f"{v:.5f}" for v in sel_vdw]
    ctok = [f"{c:.5f}" for c in sel_coul]
    tail = vtok[1:]  # the charge leg already ends at the first vdw state
    # neutral: restraint lead + vdw leg (coul stays 0)
    neutral = {"vdw-lambdas": [z] * ln + vtok}
    # charged: restraint lead + charge leg (vdw=0) + vdw leg (coul=1)
    charged = {
        "coul-lambdas": [z] * ln + ctok + ["1.00000"] * len(tail),
        "vdw-lambdas": [z] * ln + [z] * len(ctok) + tail,
    }
    if ln:
        neutral["restraint-lambdas"] = lead + [z] * len(vtok)
        charged["restraint-lambdas"] = lead + [z] * (len(ctok) + len(tail))
    return neutral, charged


def plot_schedule(legs: dict, target: float):
    """
    Save the worst-case and median acceptance against lambda together with the
    redistributed windows, and the worst-case acceptance matrix of both legs.
    :param legs: Summary per leg with its redistributed lambda values.
    :param target: Exchange acceptance to reach between adjacent windows.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, name in enumerate(LEGS):
        leg = legs[name]
        g = np.array(leg["grid"])
        ax = axes[0, col]
        ax.plot(g[:-1], np.diag(leg["worst"], 1), "o-", label="worst-case adjacent")
        ax.plot(g[:-1], leg["median"], "^--", color="gray", label="median adjacent")
        ax.axhline(target, color="r", ls="--", label=f"target {target}")
        for x in leg["lam"]:
            ax.axvline(x, color="g", alpha=0.4)
        ax.set_title(
            f"{name}: {len(g)} -> {len(leg['lam'])} windows "
            f"(worst case over {leg['n_targets']} targets)"
        )
        ax.set_xlabel(f"{name}-lambda")
        ax.set_ylabel("exchange acceptance")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
        ax2 = axes[1, col]
        im = ax2.imshow(leg["worst"], origin="lower", vmin=0, vmax=1, cmap="viridis")
        ax2.set_title(f"{name}: worst-case pairwise acceptance")
        ax2.set_xlabel("state j")
        ax2.set_ylabel("state i")
        fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.tight_layout()
    p = OUT / f"schedule_target{target:.2f}.png"
    fig.savefig(p, dpi=120)
    print(f"\nschedule plot -> {p}")


def plot_distribution(legs: dict, target: float):
    """
    Save box plots of the adjacent-pair acceptance across the targets, which show
    whether the low acceptances are outliers of single targets.
    :param legs: Summary per leg, as returned by summarize_leg.
    :param target: Exchange acceptance to reach between adjacent windows.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, name in zip(axes, LEGS):
        leg = legs[name]
        dist = leg["dist"]
        ax.boxplot(
            list(dist.T),
            showfliers=True,
            flierprops={"marker": ".", "markersize": 3},
        )
        ax.axhline(target, color="r", ls="--", label=f"target {target}")
        ax.set_xticks(range(1, dist.shape[1] + 1))
        ax.set_xticklabels(
            [f"{v:.2f}" for v in leg["grid"][: dist.shape[1]]], rotation=90, fontsize=7
        )
        ax.set_title(f"{name}: adjacent-pair acceptance across {dist.shape[0]} targets")
        ax.set_xlabel(f"{name}-lambda (lower state)")
        ax.set_ylabel("exchange acceptance")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT / "overlap_distribution.png"
    fig.savefig(p, dpi=120)
    print(f"distribution plot -> {p}")


def plot_comparison(ref: dict, legs: dict):
    """
    Plot the old against the new schedule, both the window positions of each leg and the
    full staircase of every variant.
    :param ref: Reference schedule, as returned by parse_ref_mdp.
    :param legs: Summary per leg with its redistributed lambda values.
    """
    old = build_mdp(ref, ref["coul_grid"], ref["vdw_grid"])
    new = build_mdp(ref, legs["coul"]["lam"], legs["vdw"]["lam"])
    color = {"coul": "C0", "vdw": "C1", "restraint": "C2"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    # window positions along each leg's lambda axis
    for ax, name in zip(axes[0], LEGS):
        oldg, newg = legs[name]["grid"], legs[name]["lam"]
        for x in newg:
            ax.axvline(x, color="C1", alpha=0.25)
        ax.plot(oldg, np.ones(len(oldg)), "o", color="C0", label=f"old ({len(oldg)})")
        ax.plot(newg, np.zeros(len(newg)), "s", color="C1", label=f"new ({len(newg)})")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["new", "old"])
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlabel(f"{name}-lambda")
        ax.set_title(f"{name} window positions")
        ax.legend(fontsize=8, loc="center right")
    # full schedule staircase per variant, all present lambda components vs window index
    for ax, title, o, n in (
        (axes[1, 0], "neutral", old[0], new[0]),
        (axes[1, 1], "charged", old[1], new[1]),
    ):
        keys = [k for k in MDP_KEYS if k in o]
        for key in keys:
            comp = key.split("-")[0]
            ov = [float(x) for x in o[key]]
            nv = [float(x) for x in n[key]]
            c = color[comp]
            ax.plot(range(len(ov)), ov, "o-", color=c, ms=3, lw=1, label=f"{comp} old")
            ax.plot(range(len(nv)), nv, "x--", color=c, ms=6, label=f"{comp} new")
        ax.set_title(
            f"{title}: {len(o['vdw-lambdas'])} -> {len(n['vdw-lambdas'])} windows"
        )
        ax.set_xlabel("window index")
        ax.set_ylabel("lambda")
        ax.legend(fontsize=7, ncol=len(keys))
    fig.tight_layout()
    p = OUT / "old_vs_new_schedule.png"
    fig.savefig(p, dpi=120)
    print(f"comparison plot -> {p}")


def main():
    """
    Redistribute the windows of the vdw and the charge leg to a uniform exchange
    acceptance and report the resulting schedule, its window counts and overlap plots.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target", type=float, default=0.30, help="target nearest-neighbor acceptance"
    )
    ap.add_argument(
        "--recompute", action="store_true", help="rebuild the acceptance cache"
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ref = parse_ref_mdp(REF_MDP)
    print(
        f"reference grids: vdw={len(ref['vdw_grid'])} coul={len(ref['coul_grid'])} states"
    )
    acceptance = compute_all(ref, args.recompute)
    legs = {n: summarize_leg(acceptance[n], ref[f"{n}_grid"]) for n in LEGS}
    joined = ", ".join(f"{n} over {legs[n]['n_targets']} targets" for n in LEGS)
    print(f"\naggregated worst case: {joined}")
    # distribution of overlap across all targets, per leg
    for name in LEGS:
        report_distribution(name, legs[name], args.target)
    plot_distribution(legs, args.target)
    # window count for a range of thresholds under redistribution
    print("\ntarget  vdw_windows  coul_windows")
    for t in THRESHOLDS:
        nw = [len(redistribute(legs[n]["adj"], legs[n]["grid"], t)[0]) for n in LEGS]
        print(
            f"{t:5.2f}   {nw[0]:3d} (was {len(ref['vdw_grid'])})   "
            f"{nw[1]:3d} (was {len(ref['coul_grid'])})"
        )
    # redistribution at the chosen target
    for leg in legs.values():
        leg["lam"], leg["pred"] = redistribute(leg["adj"], leg["grid"], args.target)
    neutral, charged = build_mdp(ref, legs["coul"]["lam"], legs["vdw"]["lam"])
    old_neutral, old_charged = build_mdp(ref, ref["coul_grid"], ref["vdw_grid"])
    print(f"\nchosen target acceptance = {args.target}")
    for name in LEGS:
        leg = legs[name]
        print(
            f"{name:4s}: {len(leg['grid'])} -> {len(leg['lam'])} windows "
            f"(predicted acceptance {leg['pred']:.2f})"
        )
        print(f"      {[round(float(v), 4) for v in leg['lam']]}")
    n_neutral, n_charged = len(neutral["vdw-lambdas"]), len(charged["vdw-lambdas"])
    print(
        f"\nfull schedule sizes: neutral {n_neutral} "
        f"(was {len(old_neutral['vdw-lambdas'])}), "
        f"charged {n_charged} (was {len(old_charged['vdw-lambdas'])})"
    )
    # write new mdp lambda blocks, only the components that exist
    lines = [
        f"# improved lambda schedule (Gaussian redistribution, target acceptance {args.target})",
        "",
        f"# neutral variant ({n_neutral} windows)",
    ]
    lines += [f"{k:24s} = {' '.join(neutral[k])}" for k in MDP_KEYS if k in neutral]
    lines += ["", f"# charged variant ({n_charged} windows)"]
    lines += [f"{k:24s} = {' '.join(charged[k])}" for k in MDP_KEYS if k in charged]
    sched = OUT / f"schedule_target{args.target:.2f}.mdp"
    sched.write_text("\n".join(lines) + "\n")
    print(f"\nmdp lambda blocks -> {sched}")
    plot_schedule(legs, args.target)
    plot_comparison(ref, legs)


if __name__ == "__main__":
    main()
