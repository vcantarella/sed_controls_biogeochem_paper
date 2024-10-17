# sed_controls_biogeochem_paper

Contains the scripts used to make the analysis and plots in the submitted paper: Facies and Depositional Environments of a Holocene Floodplain: Implications for Modeling Biogeochemical Reactions in Aquifers; Holdt et al. 2024

# Instructions to Install the dependencies and run the scripts

## Hardware

To run the particle tracking scripts you will need a NVIDIA GPU with CUDA support. The scripts are written specifically to the GPU. Although there is a CPU version, it is not implemented for the paper. To check the CPU version go to repository for the particle tracking scripts: [particle tracking repo](https://github.com/vcantarella/particle_track).

This readme assumes you have the GPU and the CUDA drivers installed and ready to use.

## Dependencies

The scripts are written in Python. The following packages are required to run the scripts:

### General scientific computing stack dependencies

- numpy
- matplotlib
- scipy
- numba

### Groundwater flow models (MODFLOW and FloPy)

- flopy
- MODFLOW 6 (Can be installed with flopy, see example below)

### Particle tracking

- [particle_track](https://github.com/vcantarella/particle_track) (included in this repo distribution)

### Sedimentary facies modeling

- [modified-hyvr](https://github.com/vcantarella/hyvr) (included in this repo distribution)

## Instructions to Reproduce the Results

First create a local clone of all the files in this repository:

```bash
git clone https://github.com/vcantarella/sed_controls_biogeochem_paper.git
```

### Setting up the environment

You'll need a working Python 3 environment with all the standard scientific packages installed (numpy, pandas, scipy, matplotlib, etc). The easiest (and recommended) way to get this is to download and install the [Miniconda Python distribution](https://docs.anaconda.com/miniconda/) .

Instead of manually install all the dependencies, they can all be automatically installed using a conda environment.

1. Change the directory to the cloned folder:

```bash 
cd sed_controls_biogeochem_paper
```

2. Create a new conda environment with all the dependencies:

```bash
conda env create -f environment.yml
```
3. Activate the environment:

```bash
conda activate sed_biochem
```
4. Install the modified-hyvr package:

```bash
cd hyvr
pip install -e .
cd ..
```
5. Install MODFLOW 6:

You can install manually, or run the command below (if you have already installed flopy):

```bash
get-modflow :flopy
```
6. Install CUDA dependencies:

This require that you have installed the CUDA drivers and have a compatible GPU. If you have, you can install the dependencies with the following command: (Specific details might change with different CUDA versions, refer to numba documentation for more details [numba CUDA installation](https://numba.readthedocs.io/en/stable/user/installing.html)


```bash
conda install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=12.0"
```

### Running the scripts

Go to the scripts folder:

```bash
cd scripts
```

And run the scripts in the following order:

1. Run the facies modeling script:

```bash
python s1_facies_modeling.py
```

You should now have the facies model `.npy` files

2. Run the groundwater flow model:

```bash
python s2_make_modflow_model.py
```
3. Run the particle track and cumulative TOC calculation for the facies model:

```bash
python s3_particle_track_TOC_model.py
```
4. Run the groundwater flow model and particle track for the stratified model:

 Since the stratified model is simplified, we do all the calculations in a single script. This script will run the groundwater flow model and particle track for the stratified model.

```bash
python s4_make_stratified_model.py
```

5. Analyse the results and make the plots for the paper figures:

5.1. Make the crossectional figures comparing the stratified and facies models:

```bash
python s5_model_slice_plots.py
```

5.2. Make the cumulative TOC plot:

```bash
python s6_model_outflow_analysis.py
```
### DONE!

contact me here for help
