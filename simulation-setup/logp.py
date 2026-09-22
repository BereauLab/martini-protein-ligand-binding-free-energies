"""Add the predicted logP values of the reference methods to the ligand metadata."""

import argparse
import json
import subprocess
import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parent.parent
SIMULATION_PATH = ROOT / "simulations"
ALOGPS_PATH = ROOT / "external" / "alogps"


def rdkit_canonical(s: str) -> str | None:
    """
    Convert a SMILES string to its canonical form using RDKit.
    :param s: SMILES string to canonicalize.
    :return: Canonical SMILES string, or None if it cannot be parsed.
    """
    if pd.isna(s):
        return None
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol) if mol is not None else None


def export_smiles(output_path: Path | str):
    """
    Write the neutral SMILES of every system, to be submitted to the SwissADME server.
    :param output_path: Path of the SMILES file to write.
    """
    with open(output_path, "w") as f:
        for system in SIMULATION_PATH.iterdir():
            with open(system / "data.json", "r") as data_file:
                system_data = json.load(data_file)
            neutral_smiles = system_data["neutral-smiles"]
            f.write(f"{neutral_smiles}\n")


def run_swissadme_calculations(input_path: Path | str):
    """
    Add the logP predictions of the SwissADME server to the metadata of every system,
    matched to the ligands by their canonical SMILES.
    :param input_path: SwissADME result csv file, or a directory of such files.
    """
    # Load the input data
    if Path(input_path).is_file():
        data = pd.read_csv(input_path)
    elif Path(input_path).is_dir():
        data = pd.concat(
            [
                pd.read_csv(f)
                for f in Path(input_path).glob("*.csv")
                if f.is_file() and f.suffix == ".csv"
            ],
            ignore_index=True,
        )
    else:
        raise FileNotFoundError(f"File or directory not found: {input_path}")
    # Convert the "Canonical SMILES" column to canonical form using RDKit
    data["rdkit-canonical-smiles"] = data["Canonical SMILES"].map(rdkit_canonical)
    obtained_smiles = set(data["rdkit-canonical-smiles"].dropna())
    # Iterate through the simulated systems
    missed_systems = []
    count = 0
    total_system_count = 0
    for system in SIMULATION_PATH.iterdir():
        total_system_count += 1
        if (system / "data.json").exists():
            with open(system / "data.json", "r") as f:
                system_data = json.load(f)
            smiles = rdkit_canonical(system_data.get("neutral-smiles", None))
            if smiles in obtained_smiles:
                row = data[data["rdkit-canonical-smiles"] == smiles].iloc[0]
                output_data = {
                    **system_data,
                    "xlogp3": row["XLOGP3"],
                    "wlogp": row["WLOGP"],
                    "mlogp": row["MLOGP"],
                    "silicos-it-logp": row["Silicos-IT Log P"],
                    "consensus-logp": row["Consensus Log P"],
                }
                with open(system / "data.json", "w") as f:
                    json.dump(output_data, f, indent=2)
                print(
                    f"Added SWISSADME data for ligand with SMILES {smiles} in {system.name}"
                )
                count += 1
            else:
                missed_systems.append(system)
    print(f"Added SWISSADME data for {count} out of {total_system_count} systems.")
    print(f"Missed SWISSADME data for {len(missed_systems)} systems:")
    print([s.name for s in missed_systems])


def run_alogps_calculations():
    """Add the logP predicted by the local ALOGPS tool to the metadata of every system."""
    for system in SIMULATION_PATH.iterdir():
        if (system / "data.json").exists():
            with open(system / "data.json", "r") as f:
                system_data = json.load(f)
            smiles = system_data["neutral-smiles"]
            cmd = f'./alogps-linux -s "{smiles}"'
            try:
                result = subprocess.check_output(
                    cmd, cwd=ALOGPS_PATH, shell=True, text=True
                )
                alogps_res = result.split("logP:")[1].split(")", 1)[0]
                alogp = float(alogps_res.split("(")[0])
                alogp_uncertainty = float(alogps_res.split("(")[1])
                system_data["alogp"] = alogp
                system_data["alogp-uncertainty"] = alogp_uncertainty
                system_data["neutral-smiles"] = smiles
                with open(system / "data.json", "w") as f:
                    json.dump(system_data, f, indent=2)
            except (subprocess.CalledProcessError, ValueError):
                warnings.warn(
                    f"Failed to run ALOGPS for ligand with SMILES {smiles} in {system.name}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect predicted logP values.")
    parser.add_argument(
        "mode",
        choices=["export", "alogps", "swissadme"],
        help="Which function to run",
    )
    parser.add_argument(
        "--file", type=str, help="Path to the input/output file or directory"
    )
    args = parser.parse_args()
    if args.mode == "alogps":
        run_alogps_calculations()
    elif args.file is None:
        raise ValueError(f"Please provide a file path for {args.mode} mode.")
    elif args.mode == "export":
        export_smiles(args.file)
    elif args.mode == "swissadme":
        run_swissadme_calculations(args.file)
