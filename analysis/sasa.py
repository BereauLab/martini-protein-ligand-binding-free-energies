"""Compare the solvent accessible surface of the coarse-grained and atomistic ligands."""

import json
import subprocess
from pathlib import Path

import numpy as np

RERUN = False
ROOT = Path(__file__).resolve().parent.parent
SIMULATION_PATH = ROOT / "simulations"
MDP_PATH = ROOT / "mdps" / "minimization.mdp"

for system in sorted(SIMULATION_PATH.iterdir()):
    if not (system / "ligand.itp").exists():
        continue
    sasa_path = (system / "sasa").resolve()
    sasa_path.mkdir(exist_ok=True)
    aa_sim_dir = (system / "all-atom").resolve()
    # Write vdwradii.file
    vdwradii_content = (
        "; Atomic radii for SASA calculations\n"
        "???  H     0.109\n"
        "???  C     0.175\n"
        "???  N     0.161\n"
        "???  O     0.156\n"
        "???  F     0.144\n"
        "???  S     0.179\n"
        "???  Cl    0.174\n"
        "???  Br    0.185\n"
        "???  I     0.200\n"
        "; CG beads\n"
    )
    with open(system / "ligand.itp", "r") as f:
        mode = None
        for line in f:
            if line.startswith("[") and line.strip().endswith("]"):
                mode = line.replace("[", "").replace("]", "").strip()
            elif mode == "atoms" and len(line.strip()) > 0 and not line.startswith(";"):
                parts = line.split()
                bead = parts[1]
                beadsize = "T" if bead[0] == "T" else "S" if bead[0] == "S" else "R"
                radius = {"R": 0.264, "S": 0.230, "T": 0.191}[beadsize]
                vdwradii_content += f"???  {parts[4]:>5s}   {radius:>.3f}\n"
    (sasa_path / "vdwradii.dat").write_text(vdwradii_content)
    print(f"Writing cg-min.top: {sasa_path / 'cg-min.top'}")
    if not (sasa_path / "cg-min.top").exists():
        with open(sasa_path / "cg-min.top", "w") as f:
            f.write(
                '#include "martini3.ff/martini.itp"\n'
                '#include "../ligand.itp"\n'
                "\n"
                "[ system ]\n"
                "CG-lig\n"
                "\n"
                "[ molecules ]\n"
                "ligand 1\n"
            )
    if not (sasa_path / "aa-min.top").exists():
        with open(sasa_path / "aa-min.top", "w") as f:
            f.write(
                '#include "oplsaa.ff/forcefield.itp"\n'
                '#include "../all-atom/ligand.itp"\n'
                "\n"
                "[ system ]\n"
                "AA-lig\n"
                "\n"
                "[ molecules ]\n"
                "UNL 1\n"
            )
    # Run SASA calculations for CG and all-atom structures
    # Commands must be executed in the location where vdwradii.dat is located
    if not (sasa_path / "cg-min.gro").exists() or RERUN:
        cmd_pbc = (
            f"gmx grompp -f {MDP_PATH} -c ../ligand.gro -p cg-min.top -o cg-min.tpr -po cg-min.out.mdp;"
            "gmx mdrun -deffnm cg-min && "
            "rm cg-min.edr cg-min.log cg-min.out.mdp cg-min.tpr cg-min.trr"
        )
        subprocess.run(cmd_pbc, cwd=sasa_path, shell=True, check=True)
    if not (sasa_path / "aa-min.gro").exists() or RERUN:
        cmd_pbc = (
            "gmx editconf -f ../all-atom/ligand.gro -o aa-min.gro -d 2.0 -bt cubic;"
            f"gmx grompp -f {MDP_PATH} -c aa-min.gro -p aa-min.top -o aa-min.tpr -po aa-min.out.mdp;"
            "gmx mdrun -deffnm aa-min && "
            "rm aa-min.edr aa-min.log aa-min.out.mdp aa-min.tpr aa-min.trr"
        )
        subprocess.run(cmd_pbc, cwd=sasa_path, shell=True, check=True)
    if not (sasa_path / "cg-sasa.xvg").exists() or RERUN:
        cmd1 = (
            "gmx sasa -s cg-min.gro -tv cg-density.xvg -o cg-sasa.xvg "
            "-probe 0.191 -ndots 4800 -surface 2"
        )
        subprocess.run(cmd1, cwd=sasa_path, shell=True, check=True)
    if not (sasa_path / "aa-sasa.xvg").exists() or RERUN:
        cmd2 = (
            "gmx sasa -s aa-min.gro -o aa-sasa.xvg -tv aa-density.xvg "
            "-surface 2 -probe 0.191 -ndots 4800"
        )
        subprocess.run(cmd2, cwd=sasa_path, shell=True, check=True)
    # Load SASA results
    aa_sasa_res = np.loadtxt(sasa_path / "aa-sasa.xvg", comments=("#", "@"))[1]
    cg_sasa_res = np.loadtxt(sasa_path / "cg-sasa.xvg", comments=("#", "@"))[1]
    # Load volume results
    aa_volume_res = np.loadtxt(sasa_path / "aa-density.xvg", comments=("#", "@"))[1]
    cg_volume_res = np.loadtxt(sasa_path / "cg-density.xvg", comments=("#", "@"))[1]
    # Update metadata with SASA results
    with open(system / "data.json", "r") as f:
        data = json.load(f)
    data["atom-sasa"] = aa_sasa_res
    data["cg-sasa"] = cg_sasa_res
    data["sasa-units"] = "nm^2"
    data["atom-volume"] = aa_volume_res
    data["cg-volume"] = cg_volume_res
    data["volume-units"] = "nm^3"
    with open(system / "data.json", "w") as f:
        json.dump(data, f, indent=2)
