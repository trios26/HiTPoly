# HiTPoly

A platform for setting up high throughput polymer electrolyte MD simulations.

## Installation

You can install HiTPoly directly from the source code:

```bash
# Clone the repository
git clone https://github.com/learningmatter-mit/HiTPoly.git
cd HiTPoly

# Install in editable mode
pip install -e .
```

All required dependencies (including numpy, pandas, scipy, torch, rdkit, openmm, etc.) will be automatically installed.

## Dependencies

The following main dependencies will be automatically installed:

- numpy
- pandas
- scipy
- PyTorch
- RDKit
- OpenMM
- matplotlib
- typing-extensions
- typed-argument-parser

## Requirements

To run HiTPoly, you need to:

1. Download and install LigParGen locally on your machine following the tutorial [here](https://github.com/learningmatter-mit/ligpargen)
2. Install Packmol on your workstation, [LINK](https://m3g.github.io/packmol/)
3. Install OpenBabel on your workstation `conda install openbabel -c openbabel`

HiTPoly can run simulations and interface either with Gromacs or OpenMM.

## Installation of MD Engines

### Gromacs Installation
Gromacs can be installed via package managers or built from source. For optimal performance, we recommend building from source as described on the [Gromacs website](https://manual.gromacs.org/current/install-guide/index.html)

### OpenMM Installation

Currently the simulation engine is programmed to be using CUDA. To install OpenMM with cuda run either:
`conda install -c conda-forge openmm cuda-version=12`
or
`pip install openmm[cuda12]`

To use HiTPoly with cpu compiled CUDA, platform name has to be adjusted in hitpoly/simulations/openmm_scripts.py


## Usage

Full tutorial coming soon.

## License

This project is licensed under the MIT License.
