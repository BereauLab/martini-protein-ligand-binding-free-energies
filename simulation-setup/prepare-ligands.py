"""Parametrize the benchmark ligands from atomistic reference simulations."""

import json
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from time import sleep

import numpy as np
import requests
import urllib3
from rdkit import Chem
from scipy.optimize import linear_sum_assignment

from cgbfe.file_queue import add_simulation_task_to_file_queue

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = Path(__file__).resolve().parent.parent
SIMULATION_PATH = ROOT / "simulations"
MDP_DIRECTORY = ROOT / "mdps"
AA_MINIMIZATION_MDP = MDP_DIRECTORY / "aa-minimization.mdp"
AA_EQUILIBRATION_MDP = MDP_DIRECTORY / "aa-equilibration.mdp"
AA_PRODUCTION_MDP = MDP_DIRECTORY / "aa-production.mdp"
SIMULATION_QUEUE = ROOT / "simulation-queue.dat"


def system_iterator(exclude_known_ligands: bool = True):
    """
    Iterate over all systems in the input path.
    :param exclude_known_ligands: Skip ligands with a published parametrization.
    :return: Generator yielding (system path, system metadata) tuples.
    """
    for system_path in sorted(SIMULATION_PATH.iterdir()):
        with open(system_path / "data.json", "r") as data_file:
            system_info = json.load(data_file)
        # Exclude systems with ligands with known parametrizations
        if exclude_known_ligands and system_info["known-ligand"]:
            continue
        print(f"Processing system {system_path.name}...")
        yield system_path, system_info


def get_atomistic_topologies():
    """Obtain atomistic topologies for all ligands from the LigParGen web server."""
    for system_path, system_info in system_iterator(exclude_known_ligands=False):
        all_atom_dir = system_path / "all-atom"
        all_atom_dir.mkdir(exist_ok=True)
        if (all_atom_dir / "ligand.gro").exists():
            continue
        ligand_path = system_path / "ligand.pdb"
        # Determine charge of the molecule
        mol = Chem.MolFromPDBFile(str(ligand_path), removeHs=False)
        charge = Chem.GetFormalCharge(mol)
        charge_map = {0: " 0 ", -1: " -1 ", -2: " -2 ", 1: " +1 ", 2: " +2 "}
        # Submit to LigParGen server
        with open(ligand_path, "rb") as f:
            response = requests.post(
                "https://traken.chem.yale.edu/cgi-bin/results_lpg.py",
                files={"molpdbfile": ("ligand.pdb", f, "chemical/x-pdb")},
                data={
                    "smiData": "",
                    "checkopt": " 1 ",
                    "chargetype": "cm1a",
                    "dropcharge": charge_map[charge],
                },
                verify=False,
            )
        response.raise_for_status()
        # Get download links for GRO and ITP files from the response page
        fileout_paths = re.findall(r'name="fileout"\s+value="([^"]+)"', response.text)
        gro_path = next((p for p in fileout_paths if p.endswith(".gro")), None)
        itp_path = next((p for p in fileout_paths if p.endswith(".itp")), None)
        if not gro_path or not itp_path:
            warnings.warn(
                f"Could not find GRO/ITP download links for {system_path.name}"
            )
            continue
        # Download GRO and ITP files
        download_url = "https://traken.chem.yale.edu/cgi-bin/download_lpg.py"
        for fileout, dest_name in [(gro_path, "ligand.gro"), (itp_path, "ligand.itp")]:
            dl = requests.post(
                download_url, data={"go": "go", "fileout": fileout}, verify=False
            )
            dl.raise_for_status()
            (all_atom_dir / dest_name).write_bytes(dl.content)
        sleep(1)  # Be nice to the server


def run_all_atom_reference():
    """Run the atomistic reference simulation of every ligand in water."""
    for system_path, system_info in system_iterator(exclude_known_ligands=False):
        all_atom_dir = system_path / "all-atom"
        # Place ligand in a box
        if not (all_atom_dir / "box.gro").exists():
            cmd = (
                "gmx insert-molecules -ci ligand.gro -nmol 1 "
                "-box 3.8 3.8 3.8 -rot xyz -seed 0 -o box.gro"
            )
            subprocess.run(cmd, cwd=all_atom_dir, shell=True, check=True)
        # Solvate the box
        if not (all_atom_dir / "solvated.gro").exists():
            cmd = "gmx solvate -cp box.gro -cs spc216.gro -o solvated.gro"
            subprocess.run(cmd, cwd=all_atom_dir, shell=True, check=True)
        # Create topology file
        if not (all_atom_dir / "system.top").exists():
            solvated_gro = (all_atom_dir / "solvated.gro").read_text()
            n_water = solvated_gro.count("SOL     OW")
            top_content = (
                '#include "oplsaa.ff/forcefield.itp"\n'
                '#include "ligand.itp"\n'
                '#include "oplsaa.ff/tip3p.itp"\n'
                '#include "oplsaa.ff/ions.itp"\n\n'
                "[ system ]\n; name\nLigand in Water\n\n"
                "[ molecules ]\n; compound    #mols\n"
                f"UNL    1\nSOL    {n_water}\n"
            )
            (all_atom_dir / "system.top").write_text(top_content)
        # Add ions to neutralize the system
        if not (all_atom_dir / "ions.gro").exists():
            cmd = (
                f"gmx grompp -f {AA_MINIMIZATION_MDP} -c solvated.gro -p system.top "
                "-po ions.out.mdp -o ions.tpr"
            )
            subprocess.run(cmd, cwd=all_atom_dir, shell=True, check=True)
            cmd = (
                "echo SOL | gmx genion -s ions.tpr -o ions.gro "
                "-p system.top -pname NA -nname CL -neutral"
            )
            subprocess.run(cmd, cwd=all_atom_dir, shell=True, check=True)
        # Energy minimize the system
        if not (all_atom_dir / "minimization.gro").exists():
            cmd = (
                f"gmx grompp -f {AA_MINIMIZATION_MDP} -c ions.gro -p system.top "
                "-po minimization.out.mdp -o minimization.tpr"
            )
            subprocess.run(cmd, cwd=all_atom_dir, shell=True, check=True)
            cmd = "gmx mdrun -deffnm minimization"
            subprocess.run(cmd, cwd=all_atom_dir, shell=True, check=True)
            (all_atom_dir / "minimization.edr").unlink()
        # Run an equilibration simulation
        if not (all_atom_dir / "equilibration.gro").exists():
            cmd = (
                f"gmx grompp -f {AA_EQUILIBRATION_MDP} -c minimization.gro -p system.top "
                "-po equilibration.out.mdp -o equilibration.tpr -maxwarn 1"
            )
            subprocess.run(cmd, cwd=all_atom_dir, shell=True, check=True)
            cmd = "gmx mdrun -deffnm equilibration -nt 2"
            subprocess.run(cmd, cwd=all_atom_dir, shell=True, check=True)
            (all_atom_dir / "equilibration.edr").unlink()
            (all_atom_dir / "equilibration.cpt").unlink()
        # Run a production simulation
        if not (all_atom_dir / "production.tpr").exists():
            cmd = (
                f"gmx grompp -f {AA_PRODUCTION_MDP} -c equilibration.gro -p system.top "
                "-po production.out.mdp -o production.tpr -maxwarn 1"
            )
            subprocess.run(cmd, cwd=all_atom_dir, shell=True, check=True)
            add_simulation_task_to_file_queue(
                f"{all_atom_dir.resolve()}/production", SIMULATION_QUEUE
            )


def get_ff_mapping(
    all_atom_gro: str, mapping_ndx: str, ligand_sdf: str, rmsd_threshold: float = 1.0
) -> str:
    """
    Create the content of a fast-forward mapping file from the all-atom structure and the
    mapping index. The index uses the sdf atom order while ff_map reads the atoms in the
    order of the all-atom gro file, so the indices are translated by matching the two
    structures element-wise, which requires them to be aligned.
    :param all_atom_gro: Content of the all-atom gro file.
    :param mapping_ndx: Content of the mapping index file in sdf atom order.
    :param ligand_sdf: Content of the ligand sdf file.
    :param rmsd_threshold: Maximum RMSD in Angstrom between the matched structures.
    :return: Content of the mapping file.
    """
    # All-atom GRO: atom names + coordinates in trajectory order.
    gro_lines = all_atom_gro.splitlines()
    n_atoms = int(gro_lines[1].strip())
    atom_names = [e[10:15].strip() for e in gro_lines[2 : 2 + n_atoms]]
    gro_xyz = (
        np.array(
            [
                [float(ln[20:28]), float(ln[28:36]), float(ln[36:44])]
                for ln in gro_lines[2 : 2 + n_atoms]
            ]
        )
        * 10.0  # nm -> Angstrom
    )
    # SDF: elements + coordinates in mapping.ndx order.
    mol = Chem.MolFromMolBlock(ligand_sdf, removeHs=False)
    if mol is None:
        raise ValueError("Could not parse ligand SDF for atom-order matching")
    sdf_elements = [a.GetSymbol() for a in mol.GetAtoms()]
    sdf_xyz = np.array(mol.GetConformer().GetPositions())  # Angstrom

    known_symbols = set(sdf_elements)

    def _element_from_name(name: str) -> str:
        """
        Read the element of a gro atom from its name, the leading alphabetic run. A two
        letter symbol is only accepted if it occurs in the sdf, so that names like "C0G"
        are not misread as a metal.
        :param name: Atom name of the gro file.
        :return: Element symbol of the atom.
        """
        lead = ""
        for ch in name:
            if ch.isalpha():
                lead += ch
            else:
                break
        if len(lead) >= 2 and lead[:2].capitalize() in known_symbols:
            return lead[:2].capitalize()
        return lead[:1].upper()

    gro_elements = [_element_from_name(nm) for nm in atom_names]
    if len(sdf_elements) != n_atoms:
        raise ValueError("Atom count mismatch between SDF and all-atom GRO")
    # Match SDF atom i -> GRO atom j per element (centred; translation is benign).
    # An optimal (Hungarian) assignment avoids mismatches between nearby same-element
    # atoms that a greedy nearest-neighbour search could make.
    sdf_c = sdf_xyz - sdf_xyz.mean(axis=0)
    gro_c = gro_xyz - gro_xyz.mean(axis=0)
    sdf_to_gro = {}
    for element in known_symbols:
        sdf_idx = [i for i, e in enumerate(sdf_elements) if e == element]
        gro_idx = [j for j, e in enumerate(gro_elements) if e == element]
        if len(sdf_idx) != len(gro_idx):
            raise ValueError(
                f"Element {element}: {len(sdf_idx)} atoms in SDF vs "
                f"{len(gro_idx)} in all-atom GRO"
            )
        cost = np.linalg.norm(
            sdf_c[sdf_idx][:, None, :] - gro_c[gro_idx][None, :, :], axis=2
        )
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            sdf_to_gro[sdf_idx[r]] = gro_idx[c]

    # The matched structures must be aligned for the correspondence to be valid.
    matched_gro = gro_c[[sdf_to_gro[i] for i in range(n_atoms)]]
    rmsd = float(np.sqrt(np.mean(np.sum((sdf_c - matched_gro) ** 2, axis=1))))
    if rmsd > rmsd_threshold:
        raise ValueError(
            f"SDF and all-atom GRO structures are not aligned (matched RMSD "
            f"{rmsd:.3f} A > {rmsd_threshold} A). The SDF->GRO atom correspondence "
            f"cannot be established from coordinates; align the structures (identical "
            f"orientation) before building the mapping."
        )

    # Parse mapping.ndx (SDF-order, 1-based indices).
    beads = []
    index_mapping = {}
    current_bead = None
    for line in mapping_ndx.splitlines():
        if len(line.strip()) == 0:
            continue
        elif line.startswith("[") and line.endswith("]"):
            current_bead = line[1:-1].strip()
            beads.append(current_bead)
        elif current_bead is not None:
            for idx in line.split():
                index_mapping.setdefault(int(idx), []).append(current_bead)
    if len(index_mapping) != n_atoms:
        raise ValueError("Number of mapped atoms not equal to atoms in gro")

    # Translate SDF indices -> GRO indices, then emit sorted by GRO index.
    gro_index_mapping = {}
    for sdf_idx, mapped_beads in index_mapping.items():
        gro_index_mapping[sdf_to_gro[sdf_idx - 1] + 1] = mapped_beads
    mapping = [
        f"{idx:5d} {atom_names[idx - 1]:5s} {' '.join(mapped_beads)}"
        for idx, mapped_beads in sorted(gro_index_mapping.items())
    ]

    # Add header and combine with bead list.
    content = "[ molecule ]\nUNL LIG\n\n[ martini ]\n"
    content += " ".join(beads)
    content += "\n\n[ atoms ]\n"
    content += "\n".join(mapping) + "\n"
    return content


def optimize_itp(itp_path: Path) -> None:
    """
    Cap the bond force constants of a topology and replace angles close to 180 degrees
    with a linear angle potential, both of which destabilize the simulation otherwise.
    :param itp_path: Path to the topology file, which is modified in place.
    """
    ligand_itp = itp_path.read_text().splitlines()
    # Process bonds, angles, and constraints to cap force constants
    rmode = False
    bond_lengths = {}
    for i, line in enumerate(ligand_itp):
        if line.strip().startswith((";", "#")) or len(line.strip()) == 0:
            continue
        elif line.strip().replace(" ", "") in ["[bonds]", "[angles]", "[constraints]"]:
            rmode = line.strip().replace(" ", "")
        elif line.strip().startswith("[") and line.strip().endswith("]"):
            rmode = False
        elif rmode:
            parts = line.split(";")[0].split()
            comment = line.split(";")[1] if ";" in line else ""
            if rmode == "[bonds]":
                bonds = tuple(sorted((parts[0], parts[1])))
                bond_lengths[bonds] = float(parts[3])
                # Cap bond force constants to avoid instabilities
                if float(parts[4]) > 25000:
                    comment += f" ({parts[4]})"
                    parts[4] = "25000.000"
                ligand_itp[i] = (
                    f"{parts[0]:>3s} {parts[1]:>3s} {parts[2]:>2s} "
                    f"{parts[3]:<7s} {parts[4]:>10s} ;{comment}"
                )
            elif rmode == "[constraints]":
                bonds = tuple(sorted((parts[0], parts[1])))
                bond_lengths[bonds] = float(parts[3])
            elif rmode == "[angles]":
                # Change angle function to 2 and wrap angles > 180
                theta0 = float(parts[4])
                if theta0 > 175.0:  # Replace with linear angle potential
                    bond_length1 = bond_lengths[tuple(sorted((parts[0], parts[1])))]
                    bond_length2 = bond_lengths[tuple(sorted((parts[1], parts[2])))]
                    a = bond_length2 / (bond_length1 + bond_length2)
                    comment += f" ({theta0:.3f} -> linear {bond_length1:.3f} & {bond_length2:.3f})"
                    ligand_itp[i] = (
                        f"{parts[0]:>3s} {parts[1]:>3s} {parts[2]:>3s} "
                        f"{9:>2d} {a:>7.3f} {2000.0:>10.3f} ;{comment}"
                    )
                else:
                    ligand_itp[i] = (
                        f"{parts[0]:>3s} {parts[1]:>3s} {parts[2]:>3s} "
                        f"{2:>2d} {theta0:>7.3f} {parts[5]:>10s} ;{comment}"
                    )
    itp_path.write_text("\n".join(ligand_itp))


def run_fast_forward():
    """
    Derive the bonded parameters of every ligand from its atomistic reference simulation.
    Requires the bead mapping (mapping.ndx, mapping.itp) of each system, which is created
    with the interactive mapping tool.
    """
    for system_path, system_info in system_iterator():
        all_atom_dir = system_path / "all-atom"
        if (
            not (all_atom_dir / "production.gro").exists()
            or not (system_path / "mapping.ndx").exists()
        ):
            continue
        ff_dir = system_path / "fast-forward"
        ff_dir.mkdir(exist_ok=True)
        # Fix pbc for all-atom trajectory
        if not (all_atom_dir / "production.pbc.xtc").exists():
            cmd = (
                'echo "2\n0\n" | gmx trjconv -f production.xtc -s production.tpr '
                "-o production.pbc.xtc -pbc mol -center"
            )
            subprocess.run(cmd, cwd=all_atom_dir, shell=True, check=True)
        # Create fastforward mapping file
        if not (ff_dir / "mapping.map").exists():
            all_atom_gro = (all_atom_dir / "ligand.gro").read_text()
            mapping_ndx = (system_path / "mapping.ndx").read_text()
            ligand_sdf = (system_path / "ligand.sdf").read_text()
            content = get_ff_mapping(all_atom_gro, mapping_ndx, ligand_sdf)
            (ff_dir / "mapping.map").write_text(content)
        # Map all-atom
        if not (ff_dir / "mapped.xtc").exists():
            cmd = (
                "ff_map -f all-atom/production.pbc.xtc -s all-atom/production.tpr "
                "-m fast-forward/mapping.map -o fast-forward/mapped.xtc -mols UNL"
            )
            subprocess.run(cmd, cwd=system_path, shell=True, check=True)
            (system_path / "mapped.gro").unlink(missing_ok=True)
        # Write topology file
        if not (ff_dir / "mapped.top").exists():
            content = (
                '#include "martini3.ff/martini.itp"\n'
                '#include "../mapping.itp"\n\n'
                "[ system ]\n; name\nLigand\n\n"
                "[ molecules ]\nligand    1\n"
            )
            (ff_dir / "mapped.top").write_text(content)
        # Create coarse-grained tpr file
        if not (ff_dir / "mapped.tpr").exists():
            cmd = (
                f"gmx grompp -f {AA_MINIMIZATION_MDP} -p fast-forward/mapped.top "
                "-c ligand.gro -o fast-forward/mapped.tpr -po fast-forward/temp.out.mdp"
            )
            subprocess.run(cmd, cwd=system_path, shell=True, check=True)
            (ff_dir / "temp.out.mdp").unlink()
        if not (system_path / "ligand.itp").exists():
            # Add comments to mapping.itp for ff_inter
            mapping_itp = (system_path / "mapping.itp").read_text().splitlines()
            smode = False
            comment_sections = ["[bonds]", "[angles]", "[dihedrals]"]
            scounts = {section.strip("[]"): 1 for section in comment_sections}
            for i, line in enumerate(mapping_itp):
                if line.strip().startswith((";", "#")) or len(line.strip()) == 0:
                    continue
                elif line.strip().replace(" ", "") in comment_sections:
                    smode = line.replace("[", "").replace("]", "").strip()
                elif line.strip().startswith("[") and line.strip().endswith("]"):
                    smode = False
                elif smode and ";" not in line:
                    mapping_itp[i] = line + f" ; {smode[:-1]}{scounts[smode]}"
                    scounts[smode] += 1
            (system_path / "mapping.itp").write_text("\n".join(mapping_itp))
            # Run ff_inter
            cmd = (
                "ff_inter -f mapped.xtc -s mapped.tpr -i ../mapping.itp -temperature 298 "
                "-interactions comments -dists -plots -constraints 10000000 -max-dihedral 3"
            )
            subprocess.run(cmd, cwd=ff_dir, shell=True, check=True)
            shutil.move(ff_dir / "ligand.itp", system_path / "ligand.itp")
            (ff_dir / ".mapped.xtc_offsets.npz").unlink(missing_ok=True)
            # Cap bonded and dihedral force constants to avoid instabilities
            optimize_itp(system_path / "ligand.itp")
        # Compress distribution files
        if not (ff_dir / "distributions.npz").exists():
            dist_files = list(ff_dir.glob("*.dat"))
            distribution_data = {}
            for dist_file in dist_files:
                distribution_data[dist_file.stem] = np.loadtxt(dist_file)
            np.savez_compressed(ff_dir / "distributions.npz", **distribution_data)
            for dist_file in dist_files:
                dist_file.unlink()


MODES = {
    "atomistic": get_atomistic_topologies,
    "reference": run_all_atom_reference,
    "fast-forward": run_fast_forward,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in MODES:
        print(f"Usage: python prepare-ligands.py [{'|'.join(MODES)}]")
        sys.exit(1)
    MODES[sys.argv[1]]()
