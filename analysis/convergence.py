"""Estimate the convergence of the free energies over simulation time and lambda windows."""

import json
import logging
import warnings
from pathlib import Path

logging.getLogger("pymbar").setLevel(logging.ERROR)
import numpy as np
import pandas as pd
from alchemlyb.estimators import MBAR
from pymbar.utils import ConvergenceError, DataError, ParameterError

ROOT = Path(__file__).resolve().parent.parent
SIMULATION_PATH = ROOT / "simulations"
OUTPUT = Path(__file__).parent / "convergence.json"
SYSTEMS = [
    "hsp90_10",
    "jak2_1_3E62",
    "jnk1_18627-1",
    "liga_4",
    "p38_1_1W7H",
    "t4lys_benzene",
]
LEGS = {"water": "waterS", "protein": "protein"}
REPLICAS = ["r1", "r2"]
LAMBDA_STRIDES = [0, 1, 3]
TRAJECTORY_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.00]
EQUILIBRATION = 0.05
kT_to_kcal = 0.5962


def load_leg(leg_directory: Path) -> list[pd.DataFrame]:
    """
    Load the reduced potentials of every lambda window of one leg, in lambda order.
    :param leg_directory: Directory of the leg holding the lambda windows.
    :return: Reduced potentials per window.
    """
    return [
        pd.read_pickle(window / "production.pkl.gz")
        for window in sorted(leg_directory.glob("lambda-*"))
    ]


def window_lambda(u_nk: pd.DataFrame) -> float:
    """
    Read the lambda value a window was simulated at from its index.
    :param u_nk: Reduced potentials of the window.
    :return: Lambda value of the window.
    """
    return float(u_nk.index.get_level_values(1)[0])


def kept_windows(n_windows: int, stride: int) -> list[int]:
    """
    Select the window indices left after a lambda stride, keeping both end states.
    :param n_windows: Number of windows of the leg.
    :param stride: Number of windows dropped between two kept ones.
    :return: Indices of the kept windows.
    """
    keep = list(range(0, n_windows, stride + 1))
    if keep[-1] != n_windows - 1:
        keep.append(n_windows - 1)
    return keep


def time_block(u_nk: pd.DataFrame, ratio: float, reverse: bool) -> pd.DataFrame:
    """
    Cut the leading or trailing fraction of one window, after dropping the equilibration.
    :param u_nk: Reduced potentials of the window.
    :param ratio: Fraction of the equilibrated frames to keep.
    :param reverse: Keep the trailing instead of the leading frames.
    :return: Reduced potentials of the kept frames.
    """
    equilibrated = u_nk[int(len(u_nk) * EQUILIBRATION) :]
    n_frames = max(round(len(equilibrated) * ratio), 2)
    return equilibrated.iloc[-n_frames:] if reverse else equilibrated.iloc[:n_frames]


def free_energy(
    windows: list[pd.DataFrame], keep: list[int], ratio: float, reverse: bool
) -> dict:
    """
    Estimate the free energy with MBAR over the kept windows and the given time fraction.
    :param windows: Reduced potentials of every window of the leg.
    :param keep: Indices of the windows to use.
    :param ratio: Fraction of the equilibrated frames to keep.
    :param reverse: Keep the trailing instead of the leading frames.
    :return: Free energy in kcal/mol with the covered time, sample and window counts.
    """
    reduced = [time_block(windows[index], ratio, reverse) for index in keep]
    combined = pd.concat(reduced)[[window_lambda(windows[index]) for index in keep]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mbar = MBAR().fit(combined)
    times = reduced[0].index.get_level_values("time").values
    return {
        "dG": float(mbar.delta_f_.iloc[0, -1]) * kT_to_kcal,
        "time_ns": float(times[-1] - times[0]) / 1000.0,
        "n_samples": len(combined),
        "n_windows": len(keep),
    }


results = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
for system in SYSTEMS:
    results.setdefault(system, {})
    for env, prefix in LEGS.items():
        if env in results[system]:
            print(f"{system} {env}: already in {OUTPUT.name}, skipping", flush=True)
            continue
        legs = {
            r: load_leg(SIMULATION_PATH / system / f"{prefix}-{r}") for r in REPLICAS
        }
        print(
            f"{system} {env}: "
            + ", ".join(
                f"{r} {len(w)} windows of {len(w[0])} frames" for r, w in legs.items()
            ),
            flush=True,
        )
        entry = {}
        for stride in LAMBDA_STRIDES:
            entry[str(stride)] = {}
            for ratio in TRAJECTORY_RATIOS:
                directions = {}
                for direction in ("forward", "reverse"):
                    # the whole window is both directions at once, no need to fit it twice
                    if ratio == 1.0 and direction == "reverse":
                        directions[direction] = directions["forward"]
                        continue
                    per_replica = {}
                    for replica, windows in legs.items():
                        try:
                            per_replica[replica] = free_energy(
                                windows,
                                kept_windows(len(windows), stride),
                                ratio,
                                direction == "reverse",
                            )
                        except (
                            ConvergenceError,
                            DataError,
                            ParameterError,
                            ValueError,
                            np.linalg.LinAlgError,
                        ) as error:
                            per_replica[replica] = {"dG": None, "error": str(error)}
                    directions[direction] = per_replica
                entry[str(stride)][f"{ratio:.2f}"] = directions
                reference = directions["forward"][REPLICAS[0]]
                print(
                    f"  stride {stride}, {ratio:5.0%} "
                    f"({reference.get('time_ns', 0):6.2f} ns, "
                    f"{reference.get('n_samples', 0):6d} samples, "
                    f"{reference.get('n_windows', 0)} windows): "
                    + "  ".join(
                        f"{d[:3]} "
                        + " ".join(
                            f"{v['dG']:7.3f}" if v["dG"] is not None else "     na"
                            for v in per_replica.values()
                        )
                        for d, per_replica in directions.items()
                    ),
                    flush=True,
                )
        results[system][env] = entry
        OUTPUT.write_text(json.dumps(results, indent=2))
print(f"wrote {OUTPUT}")
