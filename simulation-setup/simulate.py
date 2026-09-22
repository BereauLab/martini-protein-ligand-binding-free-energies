"""Run the coarse-grained binding and transfer free energy simulations."""

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import typer
from omegaconf import OmegaConf

from cgbfe import setup_protein_structure
from cgbfe.simulate import simulate_ligand

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


app = typer.Typer(
    help="Simulate protein systems from the Industry Benchmarks 2024 dataset",
)

ATOM_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "P": 30.974,
    "F": 18.998,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.90,
}
ROOT = Path(__file__).resolve().parent.parent
SIMULATION_PATH = ROOT / "simulations"
MDP_DIRECTORY = ROOT / "mdps"
OLIVES_SCRIPT = ROOT / "external/OLIVES/OLIVES_v2.0.1_M3.0.0.py"
MUTATIONS = {"mcl1": "HIS320:HIS"}
CHAINS = {"cdk2": "A", "cdk8": "A", "jnk1": "A", "thrombin": "H"}
DEFAULT_CONFIG = {
    "simulation": {
        "directory": str(SIMULATION_PATH),
        "gromacs": "-nt 2 -noddcheck -rdd 2.0",
        "queue": {
            "type": "file",
            "path": str(ROOT / "simulation-queue.dat"),
            "readd": False,
        },
        "martini": {
            "topology": "martini3.ff/martini.itp",
            "solvents": "martini3.ff/martini_solvents.itp",
            "ions": "martini3.ff/martini_ions.itp",
        },
        "water": {
            "composition": {"W": 1.0},
            "box_margin": 4.5,
            "neutralize": False,
            "salt": 0.0,
            "xout0": 100,
            "xout": 10000,
            "minimization": str(MDP_DIRECTORY / "minimization.mdp"),
            "equilibration": str(MDP_DIRECTORY / "solvent-equilibration.mdp"),
            "production": str(MDP_DIRECTORY / "solvent-production.mdp"),
        },
        "octanol": {
            "composition": {"OCO": 0.75, "W": 0.25 / 4},
            "box_margin": 4.5,
            "neutralize": False,
            "salt": 0.0,
            "minimization": str(MDP_DIRECTORY / "minimization.mdp"),
            "equilibration": str(MDP_DIRECTORY / "solvent-equilibration.mdp"),
            "production": str(MDP_DIRECTORY / "solvent-production.mdp"),
        },
        "protein": {
            "file": None,
            "ligand": None,
            "dummy_bead": {
                "n_virtual_sites": 8,
                "cutoff": 1.1,
                "regularization": 0.005,
            },
            "run_mode": "parallel",
            "martinize": {
                "name": "protein",
                "dssp": True,
                "ff": "martini3001",
                "ignore": "NME,ACE",
                "maxwarn": 4,
            },
            "n_reruns": 3,
            "olives": str(OLIVES_SCRIPT),
            "box_margin": 2.1,
            "salt": 0.15,
            "neutralize": True,
            "minimization": str(MDP_DIRECTORY / "minimization.mdp"),
            "equilibration": str(MDP_DIRECTORY / "protein-equilibration.mdp"),
            "production": str(MDP_DIRECTORY / "protein-production.mdp"),
            "production_charged": str(MDP_DIRECTORY / "protein-production-charged.mdp"),
        },
        "waterS": {
            "composition": {"W": 1.0},
            "box_margin": 4.5,
            "neutralize": True,
            "salt": 0.15,
            "n_reruns": 2,
            "minimization": str(MDP_DIRECTORY / "minimization.mdp"),
            "equilibration": str(MDP_DIRECTORY / "water-equilibration.mdp"),
            "production": str(MDP_DIRECTORY / "water-production.mdp"),
            "production_charged": str(MDP_DIRECTORY / "water-production-charged.mdp"),
        },
    }
}


def center_of_mass_from_sdf(sdf_path: str | Path) -> np.ndarray:
    """
    Calculate the center of mass of the molecule in an sdf file.
    :param sdf_path: Path to the sdf file.
    :return: Center of mass as a numpy array in nm.
    """
    with open(sdf_path, "r") as sdf_file:
        sdf_content = sdf_file.read()
    atoms = list(ATOM_MASSES.keys())
    atom_lines = [e.split() for e in sdf_content.splitlines()]
    atom_lines = [e for e in atom_lines if len(e) >= 4 and e[3] in atoms]
    atom_masses = np.array([ATOM_MASSES[line[3]] for line in atom_lines])
    atom_coords = np.array([line[:3] for line in atom_lines], dtype=float)
    com = np.sum(atom_coords * atom_masses[:, None], axis=0) / np.sum(atom_masses)
    return com / 10.0  # Convert from Angstrom to nm


def write_ligand_info(ligand_dir: Path):
    """
    Write the info file of a ligand with its bead types and charge, read from its topology.
    :param ligand_dir: Directory containing the ligand topology.
    """
    beads = []
    charges = []
    with open(ligand_dir / "ligand.itp", "r") as itp_file:
        read_mode = False
        for line in itp_file:
            if line.startswith((";", "#")):
                continue
            if "atoms" in line:
                read_mode = True
            elif line.startswith("[") and "]" in line:
                read_mode = False
            elif line.strip() != "" and read_mode:
                parts = line.split()
                beads.append(parts[1])
                charges.append(float(parts[6]))
    ligand_info = {
        "name": f"{ligand_dir.parent.name}-{ligand_dir.name}",
        "beads": beads,
        "charge": int(sum(charges)),
        "charged": any(c != 0 for c in charges),
    }
    with open(ligand_dir / "info.json", "w") as info_file:
        json.dump(ligand_info, info_file, indent=2)


@app.command()
def run():
    """Simulate all systems in the protein pocket and in salt water."""
    loop = asyncio.get_event_loop()
    tasks = []
    for system in sorted(SIMULATION_PATH.iterdir()):
        if not system.is_dir() or not (system / "ligand.itp").exists():
            continue
        print(f'Running simulation for system "{system.name}"...')
        # Create system info
        if not (system / "info.json").exists():
            write_ligand_info(system)
        # Setup protein structure
        config = OmegaConf.create(DEFAULT_CONFIG)
        del config.simulation.water
        del config.simulation.octanol
        config.simulation.directory = str(system)
        config.simulation.protein.file = str(system / "protein.pdb")
        target_name = system.name.split("_")[0]
        # Apply mutation if specified for the target
        mutation = MUTATIONS.get(target_name, {})
        if mutation:
            config.simulation.protein.martinize.mutate = mutation
        # Select chain if specified for the target
        chain = CHAINS.get(target_name, None)
        if chain:
            config.simulation.protein.chain = chain
        # Get ligand center of mass
        ligand_com = center_of_mass_from_sdf(system / "ligand.sdf")
        config.simulation.protein.ligand = ligand_com.tolist()
        # Setup protein structure and run simulation
        setup_protein_structure(config)
        task = simulate_ligand(ligand_directory=system, config=config)
        tasks.append(task)
    loop.run_until_complete(asyncio.gather(*tasks))


@app.command()
def logP():
    """Simulate all systems in octanol and in water to obtain their partitioning."""
    loop = asyncio.get_event_loop()
    tasks = []
    for system in sorted(SIMULATION_PATH.iterdir()):
        logP_path = system / "oco-w"
        if not logP_path.is_dir() or not (logP_path / "ligand.itp").exists():
            continue
        print(f'Running logP simulation for system "{system.name}"...')
        # Create system info
        if not (logP_path / "info.json").exists():
            write_ligand_info(logP_path)
        # Setup protein structure
        config = OmegaConf.create(DEFAULT_CONFIG)
        del config.simulation.protein
        del config.simulation.waterS
        config.simulation.directory = str(logP_path)
        # Run simulation
        task = simulate_ligand(ligand_directory=logP_path, config=config)
        tasks.append(task)
    loop.run_until_complete(asyncio.gather(*tasks))


if __name__ == "__main__":
    app()
