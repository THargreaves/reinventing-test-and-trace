## Setup Instructions

1. Create Conda environment from yaml file (`conda env create -f environment.yml`)
2. Activate environment (`conda activate test-and-trace-env`)
3. Install as Jupyter kernel (`python -m ipykernel install --user --name test-and-trace-env --display-name "Test and Trace"`)
4. Open example notebook in Jupyter using newly installed kernel

## Prototype overview
- Stan prototype 1: Setting-specific transmission rates, with underlying base rate.
- Stan prototype 2: Setting-specific transmission rates, with underlying base rate and T&T resampling.
- Stan prototype 3: Setting-specific transmission rates, with underlying base rate, T&T resampling, and false poisitive/negative test results.
- Multilevel model: 
- Tensorflow prototype: Setting-specific transmission rates, with underlying base rate.

 
