## Setup Instructions

1. Create Conda environment from yaml file (`conda env create -f environment.yml`)
2. Activate environment (`conda activate test-and-trace-env`)
3. Install as Jupyter kernel (`python -m ipykernel install --user --name test-and-trace-env --display-name "Test and Trace"`)
4. Open example notebook in Jupyter using newly installed kernel

## Prototype overview
- Base model: Models setting-specific transmission rates, with underlying base rate.
- Resampling model: Resamples data simulated for the base model into a T&T-like sample and a random survey that records activities only.
- Imperfect tests model: Keeps T&T resampling and introduces imperfect tests with false poisitive/negative results.
- Multilevel model: Introduces hierarchical structure to model class-specific transmission rates, where individual settings belong to a particular class.
- TFP prototype: Base model implemented using TensorFlow Probability.

 
