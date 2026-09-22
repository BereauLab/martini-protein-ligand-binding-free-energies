"""Count the crystal-structure interactions of every ligand with PLIP."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from plip.structure.preparation import PDBComplex
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
SIMULATION_PATH = ROOT / "simulations"
OUT = Path(__file__).resolve().parent / "plip-per-ligand.csv"
RESNAME, CHAIN, RESNUM = "LIG", "X", 999
COLS = [
    "system",
    "target",
    "hb_ldon",
    "hb_pdon",
    "hbonds_total",
    "salt_bridges",
    "water_bridges",
    "hydrophobic",
    "pi_stacking",
    "pi_cation",
]


def ligand_pdb_lines(sdf_block: str) -> list[str]:
    """
    Convert an SDF molblock to PDB HETATM lines tagged with our resname/chain/resnum.
    :param sdf_block: single-molecule SDF molblock
    :return: list of HETATM PDB record lines for the ligand
    """
    mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
    if mol is None:
        raise ValueError("bad SDF block")
    lines = []
    for line in Chem.MolToPDBBlock(mol, flavor=2).splitlines():
        if line.startswith(("HETATM", "ATOM")):
            lines.append(
                "HETATM"
                + line[6:17]
                + f"{RESNAME:>3}"
                + " "
                + CHAIN
                + f"{RESNUM:>4}"
                + line[26:]
            )
    return lines


def write_complex(protein_pdb: Path, sdf_block: str, out_path: str) -> None:
    """
    Concatenate the protein and the tagged ligand into a single PDB complex.
    :param protein_pdb: path to the apo protein PDB
    :param sdf_block: crystal-ligand SDF molblock
    :param out_path: path to write the combined complex PDB
    """
    prot = [
        ln.rstrip("\n")
        for ln in protein_pdb.read_text().splitlines()
        if not ln.startswith(("END", "MASTER", "CONECT"))
    ]
    Path(out_path).write_text(
        "\n".join(prot) + "\nTER\n" + "\n".join(ligand_pdb_lines(sdf_block)) + "\nEND\n"
    )


def plip_counts(protein_pdb: Path, sdf_block: str) -> dict:
    """
    Run PLIP on one rebuilt complex and count each interaction type.
    :param protein_pdb: path to the apo protein PDB
    :param sdf_block: crystal-ligand SDF molblock
    :return: dict of interaction counts keyed by the COLS interaction names
    """
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
        cx = tmp.name
    try:
        write_complex(protein_pdb, sdf_block, cx)
        complex = PDBComplex()
        complex.load_pdb(cx)
        complex.analyze()
        bsid = f"{RESNAME}:{CHAIN}:{RESNUM}"
        if bsid not in complex.interaction_sets:
            if not complex.interaction_sets:
                raise RuntimeError("no binding site")
            bsid = next(iter(complex.interaction_sets))
        it = complex.interaction_sets[bsid]
        return {
            "hb_ldon": len(it.hbonds_ldon),
            "hb_pdon": len(it.hbonds_pdon),
            "hbonds_total": len(it.hbonds_ldon) + len(it.hbonds_pdon),
            "salt_bridges": len(it.saltbridge_lneg) + len(it.saltbridge_pneg),
            "water_bridges": len(it.water_bridges),
            "hydrophobic": len(it.hydrophobic_contacts),
            "pi_stacking": len(it.pistacking),
            "pi_cation": len(it.pication_laro) + len(it.pication_paro),
        }
    finally:
        Path(cx).unlink(missing_ok=True)


# per-system fingerprints; the first SDF entry is the crystal ligand
rows, fails = [], []
systems = [
    s
    for s in sorted(SIMULATION_PATH.iterdir())
    if (s / "protein.pdb").exists() and (s / "ligand.sdf").exists()
]
for i, system in enumerate(systems):
    block = next(
        b for b in (system / "ligand.sdf").read_text().split("$$$$\n") if b.strip()
    )
    try:
        counts = plip_counts(system / "protein.pdb", block)
        rows.append(
            {"system": system.name, "target": system.name.split("_")[0], **counts}
        )
    except Exception as e:  # noqa: BLE001
        fails.append((system.name, str(e)[:80]))
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(systems)}", flush=True)

# write the per-ligand table
with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLS)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {OUT.name}: {len(rows)} ligands, {len(fails)} failures")
for name, err in fails:
    print("  FAIL", name, err)
