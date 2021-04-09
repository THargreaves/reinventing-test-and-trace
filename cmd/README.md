All models related to this project have been implemented in PyStan, RStan and Tensorflow, since the surrounding Python/R environments allow for elegant and readable data simulation, diagnostics and benchmarking. 

That said, these interfaces do not provide a method for pausing and resuming training. Although this did not present any issues for our smaller models, we acknowledge that this could present an issue if our solution were to be used in a practical scenario. For that reason, we have also created a demonstration of such pausable training using the more flexible shell interferace to Stan, CmdStan.

A solution like this could have be implemented using Python or R as a wrapper, but we have opted for a purely shell approach to remain entirely language agnostic.

For CmdStan installation notes, please see [here](https://mc-stan.org/docs/2_24/cmdstan-guide-2_24.pdf).