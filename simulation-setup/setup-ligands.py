"""Extract the benchmark ligands and their reference free energies into system folders."""

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.MolStandardize import rdMolStandardize

# --- SETUP ---
ROOT = Path(__file__).resolve().parent.parent
SIMULATION_PATH = ROOT / "simulations"
BENCHMARK_PATH = ROOT / "external/IndustryBenchmarks2024/industry_benchmarks"
INPUT_PATH = BENCHMARK_PATH / "input_structures" / "prepared_structures"
DG_RESULTS_PATH_FEP = (
    BENCHMARK_PATH / "analysis/schrodinger_21_4_results/combined_schrodinger_dg.csv"
)
DG_RESULTS_PATH_OPENFE = (
    BENCHMARK_PATH
    / "analysis/processed_results/combined_pymbar3_calculated_dg_data.csv"
)
SYSTEMS = [
    {"parent": "fragments", "target": "hsp90_2rings", "name": "hsp90"},
    {"parent": "fragments", "target": "hsp90_single_ring", "name": "hsp90"},
    {"parent": "fragments", "target": "jak2_set1", "name": "jak2"},
    {"parent": "fragments", "target": "jak2_set2", "name": "jak2"},
    {"parent": "fragments", "target": "liga", "name": "liga"},
    {"parent": "fragments", "target": "mcl1", "name": "mcl1"},
    {"parent": "fragments", "target": "mup1", "name": "mup1"},
    {"parent": "fragments", "target": "p38", "name": "p38"},
    {"parent": "fragments", "target": "t4_lysozyme", "name": "t4lys"},
    {"parent": "jacs_set", "target": "bace", "name": "bace"},
    {"parent": "jacs_set", "target": "jnk1", "name": "jnk1"},
    {"parent": "jacs_set", "target": "mcl1", "name": "mcl1"},
    {"parent": "jacs_set", "target": "cdk2", "name": "cdk2"},
    {"parent": "jacs_set", "target": "p38", "name": "p38"},
    {"parent": "jacs_set", "target": "thrombin", "name": "thrombin"},
    {"parent": "merck", "target": "cdk8", "name": "cdk8"},
    {"parent": "merck", "target": "cmet", "name": "cmet"},
    {"parent": "merck", "target": "eg5", "name": "eg5"},
    {"parent": "merck", "target": "hif2a", "name": "hif2a"},
]
KNOWN_CG_PATH = ROOT / "external/M3-Small-Molecules/models"
KNOWN_CG_LIST = KNOWN_CG_PATH / "table-sm3-SI.tex"
KNOWN_CG_GROS = KNOWN_CG_PATH / "gros"
KNOWN_CG_ITPS = (
    KNOWN_CG_PATH / "itps" / "opt-mono",
    KNOWN_CG_PATH / "itps" / "opt-poly",
)

# --- RUN ---


class DGResultLoader:
    """Reference free energies of the benchmark, indexed by ligand and target."""

    def __init__(self):
        self.data = {
            "fep+": pd.read_csv(DG_RESULTS_PATH_FEP),
            "openfe": pd.read_csv(DG_RESULTS_PATH_OPENFE),
        }
        for key, df in self.data.items():
            lig_col_name = {"fep+": "Ligand name", "openfe": "ligand name"}[key]
            df[lig_col_name] = (
                df[lig_col_name].str.replace(" ", "-").str.replace("_", "-")
            )

    def __call__(self, ligand_name: str, target: str, group: str) -> dict | None:
        """
        Fetch the experimental and predicted free energies of a single ligand.
        :param ligand_name: Name of the ligand as given in the sdf file.
        :param target: Name of the target the ligand belongs to.
        :param group: Name of the benchmark group the target belongs to.
        :return: Free energies of the ligand, or None if it has no unique entry.
        """
        clean_ligand_name = ligand_name.replace("_", "-").replace(" ", "-")
        selections = {}
        for name, result_data in self.data.items():
            lig_col_name = {"fep+": "Ligand name", "openfe": "ligand name"}[name]
            selection = result_data[
                (result_data[lig_col_name] == clean_ligand_name)
                & (result_data["system group"] == group)
                & (result_data["system name"] == target)
            ]
            if selection.empty:
                w = f"No {name} dG entry for ligand {ligand_name} in {target}"
                warnings.warn(w)
                return None
            elif len(selection) > 1:
                w = f"More than one {name} dG entry for ligand {ligand_name} in {target}"
                warnings.warn(w)
                return None
            selections[name] = selection
        return {
            "exp-dg": selections["fep+"].iloc[0]["Exp. dG (kcal/mol)"],
            "exp-dg-uncertainty": selections["fep+"].iloc[0][
                "Exp. dG error (kcal/mol)"
            ],
            "exp-dg-units": "kcal/mol",
            "fep-dg": selections["fep+"].iloc[0]["Pred. dG (kcal/mol)"],
            "fep-dg-uncertainty": selections["fep+"].iloc[0][
                "Pred. dG std. error (kcal/mol)"
            ],
            "fep-dg-units": "kcal/mol",
            "openfe-dg": selections["openfe"].iloc[0]["DG (kcal/mol)"],
            "openfe-dg-uncertainty": selections["openfe"].iloc[0][
                "uncertainty (kcal/mol)"
            ],
            "openfe-dg-units": "kcal/mol",
            "MMGBSA": selections["openfe"].iloc[0]["DG MMGBSA"],
        }


class KnownLigandHandler:
    """Parametrized ligands of the Martini 3 small molecule database, matched by SMILES."""

    def __init__(self):
        self.known_cgs = pd.read_csv(KNOWN_CG_LIST, sep="&", header=None)
        self.known_cgs.replace(r"(^\s+|\s+$)", "", regex=True, inplace=True)
        self.known_cgs.columns = ["short", "name", "smiles"]
        self.known_cgs["smiles"] = self.known_cgs["smiles"].apply(
            self.standardize_smiles
        )
        self.known_cgs.dropna(subset=["smiles"], inplace=True)

    def standardize_smiles(self, s: str) -> str:
        """
        Canonicalize a SMILES string of the database, dropping trailing annotations.
        :param s: SMILES string to standardize.
        :return: Canonical SMILES string, or None if it cannot be parsed.
        """
        try:
            return Chem.MolToSmiles(Chem.MolFromSmiles(s.split()[0]))
        except Exception:
            warnings.warn(f"Failed to standardize SMILES '{s.split()[0]}'")
            return None

    def __call__(self, ligand: Chem.Mol, path: Path) -> bool:
        """
        Copy the parametrization of a ligand to its system folder, if it is known.
        :param ligand: Ligand molecule to look up in the database.
        :param path: System folder to copy the topology and structure to.
        :return: Whether the ligand is part of the database.
        """
        ligand_smiles_noh = Chem.MolToSmiles(Chem.RemoveHs(ligand))
        known_cg = self.known_cgs[self.known_cgs["smiles"] == ligand_smiles_noh]
        if not known_cg.empty:
            print(ligand_smiles_noh)
            lig_itp_path = path / "ligand.itp"
            lig_gro_path = path / "ligand.gro"
            if not lig_itp_path.exists() or not lig_gro_path.exists():
                ligand_com = rdMolTransforms.ComputeCentroid(ligand.GetConformer())
                self._copy_known_ligand(known_cg.iloc[0]["short"], path, ligand_com)
            return True
        return False

    def _copy_known_ligand(
        self, ligand_name: str, output_dir: Path, center_of_mass: np.ndarray
    ):
        """
        Copy the topology and structure of a database ligand, renaming its molecule to
        'ligand' and shifting its coordinates to the given center of mass.
        :param ligand_name: Name of the ligand in the database.
        :param output_dir: System folder to copy the topology and structure to.
        :param center_of_mass: Center of mass to place the structure at, in Angstrom.
        """
        aa_itp_path = KNOWN_CG_ITPS[0] / f"{ligand_name}.itp"
        for i in range(1, len(KNOWN_CG_ITPS)):
            if aa_itp_path.exists():
                break
            aa_itp_path = KNOWN_CG_ITPS[i] / f"{ligand_name}.itp"
        # Copy ITP file
        with (
            open(aa_itp_path, "r") as f_in,
            open(output_dir / "ligand.itp", "w") as f_out,
        ):
            section_mode = False
            for line in f_in:
                if line.startswith(";") or len(line.strip()) == 0:
                    f_out.write(line)
                elif "moleculetype" in line:
                    section_mode = True
                    f_out.write(line)
                elif "[" in line and "]" in line:
                    section_mode = False
                    f_out.write(line)
                elif section_mode:
                    f_out.write(f"  ligand    {line.split()[1]}\n")
                else:
                    f_out.write(line)
        # Copy GRO file
        with open(KNOWN_CG_GROS / f"{ligand_name}.gro", "r") as f:
            structure = f.readlines()
        coordinates = np.array(
            [
                (float(e[20:28]), float(e[28:36]), float(e[36:44]))
                for e in structure[2:-1]
            ]
        )
        coordinates = coordinates - np.mean(coordinates, axis=0) + center_of_mass / 10.0
        with open(output_dir / "ligand.gro", "w") as f:
            f.write(f"{structure[0].rstrip()} (shifted)\n{len(coordinates)}\n")
            for i, line in enumerate(structure[2:-1]):
                new_line = (
                    line[:20]
                    + f"{coordinates[i, 0]:8.3f}{coordinates[i, 1]:8.3f}{coordinates[i, 2]:8.3f}\n"
                )
                f.write(new_line)
            f.write(structure[-1])


def main():
    """Create a system folder per benchmark ligand with its structure and metadata."""
    # Setup dG loader and known ligand handler
    dg_result_loader = DGResultLoader()
    known_ligand_handler = KnownLigandHandler()
    # Create output directory if it doesn't exist
    SIMULATION_PATH.mkdir(parents=True, exist_ok=True)
    # Iterate over systems and extract structures
    system_data = []
    uncharger = rdMolStandardize.Uncharger()
    for system in SYSTEMS:
        target_path = INPUT_PATH / system["parent"] / system["target"]
        # Check for cofactors
        if (target_path / "cofactors.sdf").exists():
            raise ValueError("Systems with cofactors are not supported")
        # Load and iterate over ligands
        ligands = Chem.SDMolSupplier(str(target_path / "ligands.sdf"), removeHs=False)
        for i, ligand in enumerate(ligands):
            # Check that ligand is valid
            if ligand is None:
                warnings.warn(f"Could not parse ligand {i} in {target_path}")
                continue
            # Add hydrogens if not present
            if not any(atom.GetAtomicNum() == 1 for atom in ligand.GetAtoms()):
                ligand = Chem.AddHs(ligand, addCoords=True)
            # Get ligand name and create output path
            lig_name = ligand.GetProp("_Name") if ligand.HasProp("_Name") else str(i)
            lig_name = lig_name.replace(" ", "-")
            # Check for duplicate ligands
            smiles = Chem.MolToSmiles(ligand)
            smiles_noh = Chem.MolToSmiles(Chem.RemoveHs(ligand))
            if smiles_noh in system_data:
                warnings.warn(
                    f"Duplicate ligand {lig_name} in {target_path} (SMILES: {smiles})"
                )
                continue
            # Get binding free energy data from results file
            dg_results = dg_result_loader(lig_name, system["target"], system["parent"])
            if dg_results is None:
                continue
            # Create ligand output path
            ligand_path = SIMULATION_PATH / f"{system['name']}_{lig_name}"
            ligand_path.mkdir(parents=True, exist_ok=True)
            # Copy known CG structures if available
            known_ligand = known_ligand_handler(ligand, ligand_path)
            if known_ligand:
                print(f"Found CG structure for ligand {lig_name} in {target_path}")
            # Write individual ligand sdf to file and copy protein structures
            if not (ligand_path / "ligand.pdb").exists():
                Chem.MolToPDBFile(ligand, ligand_path / "ligand.pdb")
            if not (ligand_path / "ligand.sdf").exists():
                sdf_writer = Chem.SDWriter(ligand_path / "ligand.sdf")
                sdf_writer.write(ligand)
                sdf_writer.close()
            if not (ligand_path / "protein.pdb").exists():
                shutil.copy(target_path / "protein.pdb", ligand_path / "protein.pdb")
            if (ligand_path / "data.json").exists():
                with open(ligand_path / "data.json", "r") as f:
                    existing_data = json.load(f)
                if len(existing_data) == 20:
                    print(f"Metadata exists for ligand {lig_name} in {target_path}")
                    continue
            # Save metadata
            neutral_mol = uncharger.uncharge(Chem.MolFromSmiles(smiles))
            output_data = {
                "sdf_name": lig_name,
                "smiles": smiles,
                "canonical-smiles": smiles_noh,
                "neutral-smiles": Chem.MolToSmiles(neutral_mol),
                **dg_results,
                "known-ligand": known_ligand,
            }
            system_data.append(smiles_noh)
            with open(ligand_path / "data.json", "w") as f:
                json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    main()
