# Coarse-grained absolute protein-ligand binding free energies

Code for computing absolute protein-ligand binding free energies with the Martini 3 coarse-grained force field. The simulation data itself is not part of this repository.

## Installation

```bash
pip install -e .
```

This installs the `cgbfe` package in `src`, which wraps the setup, execution and analysis of the GROMACS free energy simulations. GROMACS itself and [packmol](https://m3g.github.io/packmol) have to be available on the path.

The input structures taken from the [OpenFE benchmark](https://github.com/OpenFreeEnergy/IndustryBenchmarks2024), the [Martini 3 small molecules](https://github.com/Martini-Force-Field-Initiative/M3-Small-Molecules) and the [OLIVES script](https://github.com/Martini-Force-Field-Initiative/OLIVES) are included as submodules in `external`:

```bash
git submodule update --init
```

The logP reference values additionally need the ALOGPS command line tool from [ochem-tools](https://github.com/openochem/ochem-external-tools/tree/main/alogps) in `external/alogps`.

## Repository structure

| Path                | Content                                                          |
| ------------------- | ---------------------------------------------------------------- |
| `src/cgbfe`         | Setup, execution and free energy estimation of the simulations   |
| `simulation-setup`  | Scripts preparing and running the simulations                    |
| `mdps`              | GROMACS parameter files of all simulation stages                 |
| `analysis`          | Analysis scripts                                                 |
| `external`          | Submodules with the benchmark and the force field resources      |

The simulation data is stored in `simulations`, one directory per ligand named by `<target>_<ligand>`.

## Workflow

1. Extract the ligands from the benchmark:
```bash
python simulation-setup/setup-ligands.py
```

2. Run atomistic reference simulations:
```bash
python simulation-setup/prepare-ligands.py atomistic # Fetch topologies from LigParGen
python simulation-setup/prepare-ligands.py reference # Run simulations
```

3. Manually create mapping.ndx and mapping.itp files for the coarse-grained ligands

4. Compute bonded parameters with fast-forward:
```bash
python simulation-setup/prepare-ligands.py fast-forward
```

5. Run the binding free energy simulations:
```bash
python simulation-setup/simulate.py run
```

6. Manually create `oco-w/ligand.gro` and `oco-w/ligand.itp` in every system directory, holding the neutral form of the ligand. The structure is a copy of `ligand.gro` and the topology is `ligand.itp` with the charged beads replaced by their neutral counterparts.

7. Compute partitioning data:
```bash
python simulation-setup/simulate.py logp # Run oco-w partitioning simulations
python simulation-setup/logp.py alogps   # Compute logP with ALOGPS
python simulation-setup/logp.py export --file smiles.txt # Get SMILES list for SwissAdme
```
Process the list of SMILES with the [SwissADME web service](https://www.swissadme.ch/) and download the results as `results.csv`. Then extract the logP values:
```bash
python simulation-setup/logp.py swissadme --file results.csv # Extract logP values
```

## Simulation queue
The simulations in step 5 and 7 are appended to a file queue instead of being run directly. Start one worker per machine to work through it:

```bash
cgbfe-run-file-queue --replex 1000 --n-gpus 2 --extra "-nt 8"
```

The lambda schedules in `mdps` were derived from the state overlap of a finished set of simulations with `simulation-setup/improve-lambda.py`, which redistributes the windows to a uniform replica exchange acceptance.

