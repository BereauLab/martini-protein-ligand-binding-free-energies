"""Setup of the coarse-grained protein structure and its binding pocket dummy bead."""

import logging
import re
import shutil
from itertools import combinations
from pathlib import Path
from subprocess import run

import numpy as np
from omegaconf import ListConfig, OmegaConf
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from .utils import (
    FileUpdater,
    check_file_exists,
    clean_pdb_structure,
    get_charge_from_itp,
    get_molecule_name_from_itp,
    get_residue_com_from_pdb,
    write_json_file,
)

logger = logging.getLogger(__name__)


def add_dummy_atom_to_pocket(ligand_com: np.ndarray, sim_config: OmegaConf):
    """
    Add a dummy atom to the pocket region of the protein, used as the reference point for
    restraining the ligand. The atom is a virtual site of several backbone beads.
    :param ligand_com: Center of mass of the ligand in nm.
    :param sim_config: Simulation configuration from the config file.
    :return: Indices of the backbone beads constructing the virtual site.
    """
    logger.info("Adding dummy atom to protein pocket at %s", ligand_com)
    # Check if coarse-grained protein files exist
    protein_pdb = Path(sim_config.directory) / "protein-cg.pdb"
    protein_itp = Path(sim_config.directory) / "protein.itp"
    check_file_exists(protein_pdb, protein_itp)
    # Add dummy bead to coarse-grained pdb file
    with FileUpdater(protein_pdb) as pdb_file:
        pdb_content = pdb_file.read()
        # Look for TER line and place dummy bead there
        if pdb_content.count("TER") > 1:
            raise ValueError("PDB file must contain a single chain.")
        ter_line = re.search(r"TER.*\n", pdb_content).group(0)
        atom_id = int(ter_line[6:11])
        residue_id = int(ter_line[22:26]) + 1
        replacement = (
            f"ATOM  {atom_id:>5} DUM  DUM A{residue_id:>4}    "
            + f"{ligand_com[0] * 10:8.3f}{ligand_com[1] * 10:8.3f}{ligand_com[2] * 10:8.3f}  1.00  0.00\n"
            + f"TER   {atom_id + 1:>5}      DUM A{residue_id:>4}\n"
        )
        # Write modified pdb file
        pdb_content = pdb_content.replace(ter_line, replacement)
        pdb_file.overwrite(pdb_content)
    # Add dummy atom to itp file
    with FileUpdater(protein_itp) as itp_file:
        itp_output_lines = []
        atoms_section = False
        bb_bead_indices = []
        # Iterate through itp file to find end of atoms section
        for line in itp_file:
            if "atoms" in line and "[" in line:
                atoms_section = True
            elif line[0] in [";", "#"]:
                pass
            elif atoms_section and len(line.split()) < 5:
                atoms_section = False
                itp_output_lines.append(
                    f"{atom_id:>3} DUM  {residue_id:>3} DUM DUM {atom_id:>3}  0.0\n"
                )
            # The new Martini protein model uses CA, the old one uses BB
            elif atoms_section and (line.split()[4] == "BB" or line.split()[4] == "CA"):
                bb_bead_indices.append(int(line.split()[0]))
            itp_output_lines.append(line)
        itp_file.overwrite(itp_output_lines)
    # Calculate virtual sites for dummy atom
    ## Get backbone beads with coordinates from pdb file
    bb_bead_coords = {}
    cg_pdb_lines = [
        line.strip() for line in pdb_content.splitlines() if line.startswith("ATOM")
    ]
    for idx, line in enumerate(cg_pdb_lines):
        if idx + 1 in bb_bead_indices:
            bb_bead_coords[idx + 1] = [
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ]
    bb_bead_indices = np.array(list(bb_bead_coords.keys()))
    bb_bead_coords = np.array(list(bb_bead_coords.values())) / 10  # Convert to nm
    ## Select backbone beads within cutoff of ligand
    cutoff = sim_config.protein.dummy_bead.cutoff
    indices = np.nonzero(np.linalg.norm(bb_bead_coords - ligand_com, axis=1) < cutoff)
    selected_beads_ids = bb_bead_indices[indices[0]]
    selected_bead_coords = bb_bead_coords[indices[0]]
    logger.debug(
        "Found %d beads within %.2f nm cutoff tested for virtual site construction",
        len(selected_beads_ids),
        cutoff,
    )
    ## Find combination of backbone beads (within cutoff) that maximizes the distance between them
    pairwise_distances = cdist(selected_bead_coords, selected_bead_coords)
    best_combination = None
    max_cum_distance = 0
    n_virt_sites = sim_config.protein.dummy_bead.n_virtual_sites
    all_combinations = list(combinations(range(len(selected_beads_ids)), n_virt_sites))
    logger.debug("Testing %d combinations of beads", len(all_combinations))
    for combination in all_combinations:
        rows, cols = zip(*list(combinations(combination, 2)))
        cum_distance = np.sum(pairwise_distances[rows, cols])
        if cum_distance > max_cum_distance:
            max_cum_distance = cum_distance
            best_combination = combination
    best_bb_indices = selected_beads_ids[list(best_combination)]
    logger.debug("Best combination of beads for virtual site: %s", best_bb_indices)
    best_coordinates = selected_bead_coords[list(best_combination)]
    ## Calculate weights for each constituting bead to match the ligand COM
    n = best_coordinates.shape[0]
    lambda_reg = sim_config.protein.dummy_bead.get("regularization", 1)
    logger.debug("Using regularization param %f for virtual site weights", lambda_reg)
    # Solve constrained QP: minimise regularized residual subject to
    # sum(p) = 1 and p >= 0 (no negative weights).
    M = best_coordinates @ best_coordinates.T
    M += lambda_reg * np.eye(n) - (1 / n) * np.ones((n, n))
    c = best_coordinates @ ligand_com
    result = minimize(
        fun=lambda p: np.dot(p, M @ p) - 2 * np.dot(p, c),
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, None)] * n,
        constraints={"type": "eq", "fun": lambda p: p.sum() - 1.0},
        options={"ftol": 1e-12},
    )
    if not result.success:
        logger.warning("Weight optimisation did not converge: %s", result.message)

    params = result.x
    logger.debug(
        "Weights for virtual site: %s", ", ".join([f"{p:.3f}" for p in params])
    )
    mask = params > 0.0001
    virtual_site_coord = np.sum(
        params[mask, np.newaxis] * best_coordinates[mask], axis=0
    )
    logger.info(
        "Virtual site position: [%s], ligand COM: [%s], Distance: %.3f nm",
        ", ".join([f"{v:.3f}" for v in virtual_site_coord]),
        ", ".join([f"{v:.3f}" for v in ligand_com]),
        np.linalg.norm(virtual_site_coord - ligand_com),
    )
    n_indices = [i for i in range(n) if params[i] > 0.0001]
    vs_text = " ".join(f"{best_bb_indices[i]} {params[i]:.3f}" for i in n_indices)
    vmd = " or ".join([f"index {best_bb_indices[i]}" for i in n_indices])
    logger.debug("Virtual site definition: %s", vs_text)
    # Add virtual site to itp file
    with FileUpdater(protein_itp) as itp_file:
        itp_content = itp_file.readlines()
        for idx, line in enumerate(itp_content):
            if "virtual_sitesn" in line:
                itp_content.insert(idx + 1, f"; VMD: {vmd}\n{atom_id} 3 {vs_text}\n")
                break
        else:
            itp_content.append("\n[ virtual_sitesn ]\n")
            itp_content.append(f"{atom_id} 3 {vs_text}\n")
        itp_file.overwrite(itp_content)
    return best_bb_indices.tolist()


def setup_protein_structure(config: OmegaConf):
    """
    Setup the CG protein structure from the atomistic structure using martinize2 and
    place a dummy bead at the center of the binding pocket.
    :param config: Configuration object.
    """
    logger.info("Setting up protein structure with martinize2")
    sim_config = config.simulation
    # Setup input and output files
    input_file = sim_config.protein.file
    Path(sim_config.directory).mkdir(parents=True, exist_ok=True)
    output_clean = Path(sim_config.directory) / "protein-clean.pdb"
    output_itp = Path(sim_config.directory) / "protein.itp"
    output_top = Path(sim_config.directory) / "protein.top"
    output_pdb = Path(sim_config.directory) / "protein-cg.pdb"
    check_file_exists(input_file)
    if Path(output_itp).exists() and Path(output_pdb).exists():
        logger.info("Protein files already exist, skipping martinize step")
        return
    # Cleanup pdb structure
    if not output_clean.exists():
        chain = sim_config.protein.get("chain", None)
        clean_pdb_structure(input_file, output_clean, chain=chain)
    # Construct martinize2 command with extra arguments from config
    extra_args = ""
    if "martinize" in sim_config.protein:
        for key in sim_config.protein.martinize:
            value = sim_config.protein.martinize[key]
            if isinstance(value, bool) and value:
                extra_args += f" -{key}"
            elif not isinstance(value, bool):
                extra_args += f" -{key} {value}"
    martinize_cmd = (
        f"martinize2 -f {output_clean} -o {output_top} -x {output_pdb} {extra_args}"
    )
    # Run martinize2 command
    logger.debug("Running command: %s", martinize_cmd)
    run(martinize_cmd, shell=True, check=True)
    if not output_top.exists() or not output_pdb.exists():
        raise FileNotFoundError("Martinize2 did not produce expected output files.")
    # Move wrongly placed itp file to target location
    with open(output_top, "r", encoding="utf-8") as top_file:
        # Read name of itp file from top file
        itp_file = top_file.readlines()[2]
        itp_file = itp_file.replace("#include", "").replace('"', "").strip()
    if not Path(itp_file).exists():
        raise FileNotFoundError(f"Included itp file {itp_file} not found.")
    shutil.move(itp_file, output_itp)
    logger.debug("Moved itp file to %s", output_itp)
    # Run OLIVES if specified in config
    if "olives" in sim_config.protein:
        if not Path(sim_config.protein.olives).exists():
            raise FileNotFoundError(
                f"OLIVES script {sim_config.protein.olives} not found."
            )
        olives_cmd = (
            f"python {sim_config.protein.olives} -c {output_pdb} -i {output_itp}"
        )
        logger.debug("Running OLIVES command: %s", olives_cmd)
        run(olives_cmd, shell=True, check=True)
    # Delete useless top file
    output_top.unlink()
    logger.debug("Deleted temporary top file %s", output_top)
    # Get ligand COM from config or reference pdb file
    if isinstance(sim_config.protein.ligand, str):
        ligand_com = get_residue_com_from_pdb(input_file, sim_config.protein.ligand)
    elif isinstance(sim_config.protein.ligand, (list, ListConfig)):
        ligand_com = np.array(sim_config.protein.ligand)
    # Add dummy atom to pdb at the position of the ligand
    dummy_ref_beads = add_dummy_atom_to_pocket(ligand_com, sim_config)
    # Create protein information file
    protein_info = {
        "name": get_molecule_name_from_itp(output_itp),
        "charge": get_charge_from_itp(output_itp),
        "ligand_com": ligand_com.tolist(),
        "dummy_reference_beads": dummy_ref_beads,
    }
    write_json_file(protein_info, Path(sim_config.directory) / "protein_info.json")
