"""Report the GPU hours spent on the binding free energy simulations of every target."""

from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SIMULATION_PATH = ROOT / "simulations"
# every replica-exchange run occupies two GPUs, all lambda windows in parallel
N_GPUS = 2
LEGS = {"protein": "protein-r*", "water": "waterS-r*"}


def wall_hours(run: Path) -> float | None:
    """
    Wall time of a replica exchange run, taken from the log of its first window.
    :param run: Directory of the run holding the lambda windows.
    :return: Wall time in hours, or None if the run has not finished.
    """
    log = run / "lambda-000" / "production.log"
    if not log.exists():
        return None
    with log.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 4096))
        tail = handle.read().decode("utf-8", "replace")
    return float(tail.split("Time:")[1].split()[1]) / 3600


# per target: list over ligands of {leg: wall hours summed over repeats}
targets: dict[str, list[dict[str, float]]] = defaultdict(list)
for system in sorted(SIMULATION_PATH.iterdir()):
    ligand = {}
    for leg, pattern in LEGS.items():
        times = [wall_hours(run) for run in sorted(system.glob(pattern))]
        ligand[leg] = sum(t for t in times if t is not None)
        ligand[f"{leg}-runs"] = [t for t in times if t is not None]
    if ligand["protein"]:
        targets[system.name.split("_", 1)[0]].append(ligand)

targets = dict(
    sorted(targets.items(), key=lambda kv: np.median([l["protein"] for l in kv[1]]))
)

# GPU hours per single repeat of each leg, and per ligand summed over all repeats
header = (
    f"{'target':10s} {'lig':>4s} {'prot/run':>9s} {'wat/run':>8s} "
    f"{'prot':>6s} {'wat':>5s} {'1 repeat':>8s} {'all rep':>8s}"
)
print(header)
print("-" * len(header))
for target, ligands in targets.items():
    protein = np.mean([t for l in ligands for t in l["protein-runs"]]) * N_GPUS
    water = np.mean([t for l in ligands for t in l["water-runs"]]) * N_GPUS
    total = np.mean([l["protein"] + l["water"] for l in ligands]) * N_GPUS
    print(
        f"{target:10s} {len(ligands):4d} {protein / N_GPUS * 60:6.1f} min {water / N_GPUS * 60:4.1f} min "
        f"{protein:6.2f} {water:5.2f} {protein + water:8.2f} {total:8.2f}"
    )

all_runs = [
    t * 60 for ligands in targets.values() for l in ligands for t in l["protein-runs"]
]
all_water = [
    t * 60 for ligands in targets.values() for l in ligands for t in l["water-runs"]
]
single = [
    np.mean([t for l in ligands for t in l["protein-runs"]])
    + np.mean([t for l in ligands for t in l["water-runs"]])
    for ligands in targets.values()
]
single = [s * N_GPUS for s in single]
per_target = [
    np.mean([(l["protein"] + l["water"]) * N_GPUS for l in ligands])
    for ligands in targets.values()
]
print()
print(
    f"mean protein run: {np.mean(all_runs):.1f} min, mean water run: {np.mean(all_water):.1f} min"
)
print(
    f"GPU hours per ligand, one repeat: mean {np.mean(single):.2f} "
    f"(min {np.min(single):.2f}, max {np.max(single):.2f})"
)
print(
    f"GPU hours per ligand, all repeats: mean {np.mean(per_target):.2f} "
    f"(min {np.min(per_target):.2f}, max {np.max(per_target):.2f})"
)
