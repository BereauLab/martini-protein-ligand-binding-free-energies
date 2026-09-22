"""Data loading, descriptors and figure style for the analysis notebook."""

import json
import math
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import freesasa
import matplotlib as mpl
import MDAnalysis as mda
import numpy as np
import pandas as pd
import vermouth.forcefield
from fast_forward.interaction_distribution import interaction_distribution
from fast_forward.itp_parser_sub import read_itp
from fast_forward.itp_to_ag import ITPInteractionMapper
from fast_forward.score import calc_score
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ticker import MaxNLocator
from MDAnalysis import transformations as trans
from openmm import NonbondedForce
from openmm.app import ForceField, Modeller, PDBFile
from openmm.unit import angstrom
from rdkit import Chem, RDLogger
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

RDLogger.DisableLog("rdApp.*")
freesasa.setVerbosity(freesasa.silent)

ROOT = Path(__file__).resolve().parent.parent
SIMULATION_PATH = ROOT / "simulations"
PLIP_CSV = Path(__file__).parent / "plip-per-ligand.csv"
LOGP_REF_KEYS = ["xlogp3", "mlogp", "silicos-it-logp", "alogp"]
EXTRA_KEYS = ["MMGBSA", "atom-sasa", "cg-sasa"]
# PLIP interaction counts kept for the ranking analysis (columns of PLIP_CSV). hb_ldon /
# hb_pdon are H-bonds with the ligand as donor / acceptor (PLIP "protein donor" = ligand
# acceptor); hbonds_total is their sum.
PLIP_KEYS = [
    "hb_ldon",
    "hb_pdon",
    "hbonds_total",
    "salt_bridges",
    "water_bridges",
    "hydrophobic",
    "pi_stacking",
    "pi_cation",
]
# shortest series for which a within-target rank correlation is meaningful
MIN_LIGANDS = 5
KYTE_DOOLITTLE = {
    "ILE": 4.5,
    "VAL": 4.2,
    "LEU": 3.8,
    "PHE": 2.8,
    "CYS": 2.5,
    "MET": 1.9,
    "ALA": 1.8,
    "GLY": -0.4,
    "THR": -0.7,
    "SER": -0.8,
    "TRP": -0.9,
    "TYR": -1.3,
    "PRO": -1.6,
    "HIS": -3.2,
    "GLU": -3.5,
    "GLN": -3.5,
    "ASP": -3.5,
    "ASN": -3.5,
    "LYS": -3.9,
    "ARG": -4.5,
}
PROTEIN_FORCEFIELD = ForceField("amber14-all.xml")
_PROTEIN_RESIDUES = set(KYTE_DOOLITTLE) | {"ACE", "NME"}
_AROMATIC = {"PHE", "TYR", "TRP", "HIS"}
_POLAR = {"SER", "THR", "ASN", "GLN", "CYS", "TYR", "HIS", "ASP", "GLU", "LYS", "ARG"}
_NONPOLAR = {"ALA", "VAL", "LEU", "ILE", "MET", "PRO", "GLY", "PHE", "TRP"}

# Text block geometry: letterpaper, 0.75in side margins, 0.25in columnsep.
COLUMN_WIDTH = 3.375
TEXT_WIDTH = 7.0
FIGURE_DIR = ROOT / "figures"
# Body text is 10pt; figure text sits 1-2pt below it.
BASE_SIZE = 8.0
FONT = "Nimbus Sans"
COLORS = [
    "#3362b0",
    "#cc3164",
    "#1ea69c",
    "#f78746",
    "#a196ef",
    "#4bcefd",
    "#ffc07b",
    "#628b28",
    "#e484bf",
    "#8a4096",
    "#0cdbb6",
    "#0e89e5",
    "#58ba59",
    "#e35019",
    "#bfc544",
]
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": [FONT],
    "font.size": BASE_SIZE,
    "mathtext.fontset": "stixsans",
    "axes.labelsize": BASE_SIZE + 1,
    "xtick.labelsize": BASE_SIZE,
    "ytick.labelsize": BASE_SIZE,
    "legend.fontsize": BASE_SIZE,
    "legend.frameon": False,
    "axes.prop_cycle": mpl.cycler(color=COLORS),
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 3.5,
    "hatch.linewidth": 0.3,
    "pdf.fonttype": 42,
    "savefig.dpi": 600,
    "savefig.bbox": "standard",
}


def logP_to_dG(logP: float) -> float:
    """
    Convert logP to free energy of transfer in kcal/mol at 300 K
    :param logP: logP value
    :return: free energy of transfer in kcal/mol
    """
    return logP * 1.9872 / 1000 * 300 * np.log(10)


def iterate_targets(data: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    """
    Iterate over unique targets in the data and yield a tuple of (target, target_data)
    :param data: DataFrame containing the data with a "target" column
    :return: generator yielding (target, target_data) tuples
    """
    for target in data["target"].unique():
        yield target, data[data["target"] == target]


def load_ligand_data() -> pd.DataFrame:
    """
    Load ligand data from the simulations directory and return a DataFrame with the relevant
    information.
    :return: DataFrame containing ligand data
    """
    dG_data = []
    for system in SIMULATION_PATH.iterdir():
        if not (system / "info.json").exists():
            continue
        with open(system / "info.json", "r") as f:
            info = json.load(f)
        protein_dG = [
            info[f"protein-r{i}"]["dG"] for i in range(1, 4) if f"protein-r{i}" in info
        ]
        protein_ddG = (  # standard error of the mean for protein dG
            np.std(protein_dG) / np.sqrt(len(protein_dG))
            if len(protein_dG) > 1
            else 0.0
        )
        protein_dG = np.mean(protein_dG)
        water_dG = [
            info[f"waterS-r{i}"]["dG"] for i in range(1, 3) if f"waterS-r{i}" in info
        ]
        water_ddG = (  # standard error of the mean for water dG
            np.std(water_dG) / np.sqrt(len(water_dG)) if len(water_dG) > 1 else 0.0
        )
        water_dG = np.mean(water_dG)
        with open(system / "data.json", "r") as f:
            data = json.load(f)
        mol = Chem.MolFromSmiles(data["canonical-smiles"])
        with open(system / "oco-w" / "info.json", "r") as f:
            ocow_info = json.load(f)
        dG_data.append(
            {
                "system": system.name,
                "target": system.name.split("_")[0],
                "ligand": system.name.split("_", 1)[1],
                "smiles": data["canonical-smiles"],
                "charged": info["charged"],
                "charge": info["charge"],
                "n_heavy": mol.GetNumHeavyAtoms() if mol is not None else np.nan,
                **{key: logP_to_dG(data[key]) for key in LOGP_REF_KEYS},
                "logp": float(np.mean([data[key] for key in LOGP_REF_KEYS])),
                "dG": water_dG - protein_dG,
                "ddG": np.sqrt(water_ddG**2 + protein_ddG**2),
                "exp": data["exp-dg"],
                "exp-err": data["exp-dg-uncertainty"],
                "fep": data["fep-dg"],
                "fep-err": data["fep-dg-uncertainty"],
                "openfe-dg": data["openfe-dg"],
                "openfe-err": data["openfe-dg-uncertainty"],
                **{key: data[key] for key in EXTRA_KEYS if key in data},
                "oco-w-dG": ocow_info["octanol"]["dG"] - ocow_info["water"]["dG"],
                "oco-w-ddG": np.sqrt(
                    ocow_info["octanol"]["ddG"] ** 2 + ocow_info["water"]["ddG"] ** 2
                ),
            }
        )
    dG_data = pd.DataFrame(dG_data)
    return dG_data


def load_ligand_features() -> pd.DataFrame:
    """
    Per-ligand frame for the ranking analysis: the CG/FEP/experimental binding data from
    load_ligand_data (which already carries the mean predicted logP as lipophilicity, plus
    n_heavy for size and the formal charge) merged with the PLIP crystal-complex interaction
    counts (PLIP_KEYS).
    :return: DataFrame with one row per ligand
    """
    data = load_ligand_data()
    plip = pd.read_csv(PLIP_CSV)
    return data.merge(plip[["system", *PLIP_KEYS]], how="inner", on="system")


def _series_spearman(
    potency: pd.Series, prop: pd.Series, zero_fill: bool = False
) -> float:
    """
    Signed within-series Spearman correlation of potency with a ligand property.
    :param potency: reference potency per ligand (-dG, higher is stronger)
    :param prop: ligand property per ligand
    :param zero_fill: if the property has no within-series variance, return 0.0 (the series
        does not probe that axis) instead of nan; used for the sparse interaction counts
    :return: Spearman rho, or nan when the series is shorter than MIN_LIGANDS, the potency
        is constant, or the property is constant (unless zero_fill)
    """
    potency, prop = np.asarray(potency, float), np.asarray(prop, float)
    mask = ~(np.isnan(potency) | np.isnan(prop))
    potency, prop = potency[mask], prop[mask]
    if len(potency) < MIN_LIGANDS or np.std(potency) < 1e-9:
        return math.nan
    if np.std(prop) < 1e-9:
        return 0.0 if zero_fill else math.nan
    return spearmanr(potency, prop).statistic


# ranking descriptors: sar_X = signed Spearman(FEP potency, property); zero_fill marks the
# sparse interaction counts that read 0 (not nan) when a series does not vary that channel.
RANKING_DESCRIPTORS = {
    "sar_logp": ("logp", False),
    "sar_size": ("n_heavy", False),
    "sar_charge": ("charge", True),
    "sar_hb": ("hbonds_total", True),
    "sar_hb_don": ("hb_ldon", True),
    "sar_hb_acc": ("hb_pdon", True),
    "sar_salt": ("salt_bridges", True),
    "sar_water": ("water_bridges", True),
    "sar_hydrophobic": ("hydrophobic", True),
    "sar_pi_stacking": ("pi_stacking", True),
    "sar_pi_cation": ("pi_cation", True),
}


def ranking_features() -> pd.DataFrame:
    """
    Per-target ranking descriptors. The response rho = Spearman(CG dG, FEP dG) says whether
    CG orders the series like FEP, and each sar_X = signed Spearman(FEP potency, property X)
    how strongly the potency within the series is driven by that property.
    :return: DataFrame with one row per target (targets with >= MIN_LIGANDS ligands)
    """
    data = load_ligand_features()
    rows = []
    for target, g in iterate_targets(data):
        if len(g) < MIN_LIGANDS:
            continue
        potency = -g["fep"]
        row = {
            "target": target,
            "n": len(g),
            "rho": _series_spearman(g["fep"], g["dG"]),
        }
        for name, (col, zero_fill) in RANKING_DESCRIPTORS.items():
            row[name] = _series_spearman(potency, g[col], zero_fill=zero_fill)
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _fisher_bounds(r: float, se_z: float) -> tuple[float, float]:
    """
    +/-1 SE bounds of a correlation formed on the Fisher z-scale and mapped back with tanh, so
    the bounds always stay within (-1, 1): (tanh(z - se_z), tanh(z + se_z)) with z = atanh(r).
    r is clamped off +/-1 for the atanh; the returned bounds are widened to bracket the original
    r so a perfect |r| = 1 gives a one-sided band rather than a bound on the wrong side of r.
    :param r: correlation coefficient
    :param se_z: standard error of the z-transformed correlation
    :return: (lower, upper) one-SE bounds in correlation units, with lower <= r <= upper
    """
    z = math.atanh(max(-0.999999, min(0.999999, r)))
    return min(math.tanh(z - se_z), r), max(math.tanh(z + se_z), r)


def spearman_se(r: float, n: int) -> tuple[float, float]:
    """
    One-standard-error bounds of a Spearman correlation via the Fisher z-transform, using the
    Bonett-Wright non-null variance Var(z) = (1 + r**2 / 2) / (n - 3). A +/-1 SE band, not a CI.
    :param r: Spearman correlation
    :param n: number of paired observations
    :return: (lower, upper) one-SE bounds in correlation units (nan, nan for n < 4)
    """
    if n < 4:
        return math.nan, math.nan
    return _fisher_bounds(r, math.sqrt((1 + r**2 / 2) / (n - 3)))


def kendall_se(tau: float, n: int) -> tuple[float, float]:
    """
    One-standard-error bounds of Kendall's tau via the Fisher z-transform, using the
    Fieller-Hartley-Pearson variance Var(z) = 0.437 / (n - 4). A +/-1 SE band, not a CI.
    :param tau: Kendall rank correlation
    :param n: number of paired observations
    :return: (lower, upper) one-SE bounds in correlation units (nan, nan for n < 5)
    """
    if n < 5:
        return math.nan, math.nan
    return _fisher_bounds(tau, math.sqrt(0.437 / (n - 4)))


def use_paper_style():
    """Apply the rcParams of the paper figures and raise if FONT is not installed."""
    findfont(FontProperties(family=FONT), fallback_to_default=False)
    mpl.rcParams.update(RC)


def figure(
    width: Literal["single", "double"] = "single",
    height: float | None = None,
    subplots: bool = True,
    **kwargs,
):
    """
    Create a figure sized for one column or the full text block.
    :param width: "single" for one column, "double" for the full text block.
    :param height: Height in inches, defaults to 0.62 * COLUMN_WIDTH.
    :param subplots: Whether to return a single subplot or a grid of subplots.
    :param kwargs: Extra kwargs go to plt.subplots or plt.figure, e.g. nrows/ncols/sharex.
    :return: Figure and axes, or only the figure if subplots is False.
    """
    w = COLUMN_WIDTH if width == "single" else TEXT_WIDTH
    h = height if height is not None else COLUMN_WIDTH * 0.62
    if subplots:
        return plt.subplots(figsize=(w, h), layout="constrained", **kwargs)
    return plt.figure(figsize=(w, h), layout="constrained", **kwargs)


def save_figure(fig: Figure, name: str) -> Path:
    """
    Write FIGURE_DIR/<name>.pdf at the exact figure size.
    :param fig: matplotlib Figure object.
    :param name: Filename without extension.
    :return: Path to the saved figure.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.pdf"
    fig.savefig(path)
    return path


def square_with_diagonal(
    ax: plt.Axes, pad: float = 0, nticks: int | str = "auto", **kwargs
):
    """
    Make the axes square and add a diagonal line.
    :param ax: matplotlib Axes object
    :param pad: Padding to add around the data points.
    :param nticks: Number of ticks for the axes.
    :param kwargs: Extra kwargs go to ax.plot for the diagonal line.
    """
    lim = (
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    )
    lim = (lim[0] - pad * (lim[1] - lim[0]), lim[1] + pad * (lim[1] - lim[0]))
    ax.set(xlim=lim, ylim=lim)
    ax.plot(lim, lim, color="black", lw=0.5, zorder=0, **kwargs)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nticks))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nticks))


def _pocket_sasa(
    pdb_path: str, model: Modeller, pocket: list[int]
) -> tuple[float, float]:
    """
    Burial and apolar surface of the pocket residues on the apo (ligand-free) protein.
    :param pdb_path: protein PDB path
    :param model: OpenMM Modeller whose residue indices match pocket
    :param pocket: residue indices lining the pocket
    :return: (mean relative SASA over pocket residues, apolar fraction of pocket SASA)
    """
    sasa_options = {
        "hetatm": False,
        "hydrogen": False,
        "skip-unknown": True,
        "halt-at-unknown": False,
    }
    areas = freesasa.calc(
        freesasa.Structure(str(pdb_path), options=sasa_options)
    ).residueAreas()
    key = {r.index: (r.chain.id, r.id) for r in model.topology.residues()}
    rel, apolar, total = [], 0.0, 0.0
    for i in pocket:
        c, n = key[i]
        if c in areas and n in areas[c]:
            ra = areas[c][n]
            apolar += ra.apolar
            total += ra.total
            if ra.hasRelativeAreas and not math.isnan(ra.relativeTotal):
                rel.append(ra.relativeTotal)
    mean_rel = float(np.mean(rel)) if rel else math.nan
    apolar_frac = apolar / total if total else math.nan
    return mean_rel, apolar_frac


def pocket_features(system_dir: str, cutoff: float = 6.0) -> dict:
    """
    Structure-only descriptors of the pocket lining, with the pocket defined as the
    protein residues holding a heavy atom within cutoff of the crystal ligand.
    :param system_dir: system directory holding protein.pdb and ligand.sdf
    :param cutoff: pocket radius in Angstrom around the ligand
    :return: Residue counts of the pocket lining (n_res, n_charged, n_polar, n_aromatic,
        n_nonpolar, the classes overlap), their fractions over n_res, the net charge and
        the mean Kyte-Doolittle hydropathy kd; the SASA of the apo pocket as enclosure
        (mean relative SASA, low is buried) and apolar (apolar fraction of the surface);
        and within the ligand shell iface (heavy atoms), packing (heavy atoms within
        4.5 A per ligand atom), hbond (polar atoms), donor and acceptor
    """
    system_dir = Path(system_dir)
    lig = (
        Chem.MolFromMolFile(str(system_dir / "ligand.sdf"), removeHs=True)
        .GetConformer()
        .GetPositions()
    )
    pdb = PDBFile(str(system_dir / "protein.pdb"))
    model = Modeller(pdb.topology, pdb.positions)
    for residue in model.topology.residues():
        if (
            residue.name == "NMA"
        ):  # C-terminal cap; amber14 knows it as NME (matched by bond graph)
            residue.name = "NME"
    model.delete(
        [r for r in model.topology.residues() if r.name not in _PROTEIN_RESIDUES]
    )
    system = PROTEIN_FORCEFIELD.createSystem(model.topology)
    force = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    atoms = list(model.topology.atoms())
    q = np.array([force.getParticleParameters(i)[0]._value for i in range(len(atoms))])
    pos = np.array(model.positions.value_in_unit(angstrom))
    sym = np.array([a.element.symbol for a in atoms])
    rid = np.array([a.residue.index for a in atoms])
    heavy, polar = sym != "H", np.isin(sym, ("N", "O"))
    dist = cdist(pos[heavy], lig)  # heavy protein atoms to ligand atoms
    near = np.zeros(len(atoms), bool)
    near[heavy] = dist.min(1) < cutoff
    pocket = sorted(set(rid[near].tolist()))
    resname = {r.index: r.name for r in model.topology.residues()}
    shell = near & polar  # polar shell atoms for the donor/acceptor split
    hpos = pos[sym == "H"]
    has_h = (
        (cdist(pos[shell], hpos) < 1.2).any(1)
        if hpos.size and shell.any()
        else np.zeros(int(shell.sum()), bool)
    )
    enclosure, apolar = _pocket_sasa(system_dir / "protein.pdb", model, pocket)
    charge_res = {i: round(q[rid == i].sum()) for i in pocket}  # net charge per residue
    n_res = len(pocket)
    n_charged = int(sum(charge_res[i] != 0 for i in pocket))
    n_polar = sum((resname[i] in _POLAR) and (charge_res[i] == 0) for i in pocket)
    n_nonpolar = sum(resname[i] in _NONPOLAR for i in pocket)
    n_aromatic = sum(resname[i] in _AROMATIC and (charge_res[i] == 0) for i in pocket)
    kd = float(
        np.mean(
            [KYTE_DOOLITTLE[resname[i]] for i in pocket if resname[i] in KYTE_DOOLITTLE]
        )
    )
    return {
        # amino-acid composition: counts, fractions over n_res, net charge, hydropathy
        "n_res": n_res,
        "n_charged": n_charged,
        "n_polar": n_polar,
        "n_aromatic": n_aromatic,
        "n_nonpolar": n_nonpolar,
        "f_charged": n_charged / n_res,
        "f_polar": n_polar / n_res,
        "f_aromatic": n_aromatic / n_res,
        "f_nonpolar": n_nonpolar / n_res,
        "charge": int(sum(charge_res.values())),
        "kd": kd,
        # SASA of the apo pocket
        "enclosure": enclosure,
        "apolar": apolar,
        # atom-count based, within the ligand shell
        "iface": int(near.sum()),
        "packing": float((dist < 4.5).sum(0).mean()),
        "hbond": int(shell.sum()),
        "donor": int(has_h.sum()),
        "acceptor": int(((sym[shell] == "O") | (~has_h & (sym[shell] == "N"))).sum()),
    }


def load_pocket_features(cutoff: float = 6.0) -> pd.DataFrame:
    """
    Pocket descriptors for every simulation system.
    :param cutoff: pocket radius in Angstrom around the ligand
    :return: DataFrame with system, target and the pocket_features columns
    """
    rows = [
        {"system": s.name, "target": s.name.split("_")[0], **pocket_features(s, cutoff)}
        for s in sorted(SIMULATION_PATH.iterdir())
        if (s / "ligand.sdf").exists() and (s / "protein.pdb").exists()
    ]
    return pd.DataFrame(rows)


def score_bonded_distributions(
    systems: Iterable[str] | None = None,
    cg_dir: str = "oco-w/water/lambda-000",
    ref_npz: str = "fast-forward/distributions.npz",
    itp_files: tuple = ("mapping.itp",),
    tpr_name: str = "production.tpr",
    traj_name: str = "production.xtc",
    hellinger_weight: float = 0.7,
) -> dict:
    """
    ff_assess-style bonded-distribution scores per system. The CG bonds, angles and dihedrals
    are histogrammed from the CG trajectory and compared to the ff_inter atomistic reference
    with calc_score (weighted Hellinger distance plus normalised mean shift, lower is closer).
    :param systems: system names to score; None scores every system with the required files
    :param cg_dir: CG simulation dir holding tpr/traj, relative to each system
    :param ref_npz: ff_inter atomistic distributions npz, relative to each system
    :param itp_files: CG topology itp(s) defining the interaction groups, per system
    :param tpr_name: CG run tpr inside cg_dir
    :param traj_name: CG trajectory inside cg_dir
    :param hellinger_weight: Hellinger weight in calc_score (ff_assess default 0.7)
    :return: {system: {"bonds": [...], "angles": [...], "dihedrals": [...]}} of per-group scores
    """
    inter_types = ("bonds", "angles", "dihedrals")
    dirs = (
        [SIMULATION_PATH / s for s in systems]
        if systems is not None
        else sorted(SIMULATION_PATH.iterdir())
    )
    all_scores = {}
    for system in dirs:
        cg = system / cg_dir
        missing = [p for p in (system / ref_npz, cg / traj_name) if not p.exists()]
        if missing:
            print(
                f"{system.name}: skipping, missing {', '.join(str(p.relative_to(system)) for p in missing)}"
            )
            continue
        try:
            ref = np.load(system / ref_npz, allow_pickle=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u = mda.Universe(str(cg / tpr_name), str(cg / traj_name))
                u.trajectory.add_transformations(trans.unwrap(u.atoms))
                u.transfer_to_memory()  # interaction_distribution reads the coordinate array
            ff = vermouth.forcefield.ForceField("dummy")
            for itp in itp_files:
                with open(system / itp) as fh:
                    read_itp(fh.readlines(), ff)
            mapper = ITPInteractionMapper(u, ff.blocks.values(), ff.blocks.keys())
            scores = {t: [] for t in inter_types}
            for molname in ff.blocks:
                groups = mapper.get_interactions_group(molname)[0]
                for inter_type in inter_types:
                    for group_name, pair_idxs in groups[inter_type].items():
                        key = f"{group_name}_{inter_type}_distr"
                        if key not in ref:
                            continue
                        result = interaction_distribution(
                            u, inter_type, pair_idxs, group_name
                        )
                        cg_probs = (
                            result[0] if isinstance(result, tuple) else result
                        ).T[1]
                        scores[inter_type].append(
                            float(
                                calc_score(
                                    ref[key].T[1],
                                    cg_probs,
                                    [hellinger_weight, 1 - hellinger_weight],
                                    inter_type,
                                )
                            )
                        )
            all_scores[system.name] = scores
        except Exception as exc:  # noqa: BLE001
            print(f"{system.name}: failed ({exc})")
    return all_scores


def repeat_spread(prefix: str) -> np.ndarray:
    """
    Standard error over the repeats of one leg, for every ligand that has at least two.
    :param prefix: Name of the leg, without the repeat suffix.
    :return: Standard error of the mean per ligand, in kcal/mol.
    """
    values = []
    for system in sorted(SIMULATION_PATH.iterdir()):
        if not (system / "info.json").exists():
            continue
        with open(system / "info.json") as handle:
            info = json.load(handle)
        repeats = [
            info[f"{prefix}-r{i}"]["dG"]
            for i in range(1, 10)
            if f"{prefix}-r{i}" in info
        ]
        if len(repeats) >= 2:
            values.append(np.std(repeats, ddof=1) / np.sqrt(len(repeats)))
    return np.array(values)
