# Reinventing Test and Trace

**_A Bayesian Approach For Estimating SARS-CoV-2 Setting-Specific Transmission Rates_**

## Abstract

In a recent damning report by the National Audit Office, it was revealed that the UK government's Test and Trace  programme to  combat the Coronavirus pandemic, repeatedly failed to  meet targets for contract tracing and test results, despite escalating costs of over £22 million. One such failing was the lack of data collection and analysis aimed at understanding setting-specific SARS-CoV-2 transmission, thus preventing the implementation of effective, data-driven policy. This has resulted in the grouping of  disparate  activities  when  determining  lockdown  rules, that  do  not share  remotely  similar transmission rates according to the few small-scale studies that exist.

In contrast, we demonstrate the use of an alternative methodology in which recipients of antigen tests complete a short survey detailing the activities they recently participated in. These are fed into a novel, first-principles-based Bayesian model, to infer where transmission is  occurring. We benchmark the performance of the solution on simulated data in terms of convergence and runtime as the number of settings and  observations increase,  whilst  evaluating the  impact  of  model  priors  derived from expert opinion on our inference. Finally, we introduce a hierarchical  variant  of the standard model that allows us to incorporate interventions such as social distancing and sanitisation into our model and intuit their effectiveness in each respective setting.

Our model is implemented in Python, comparing the use of three statistical computing frameworks (Stan, TensorFlow Probability, and PyMC3) to determine the most computationally efficient approach. Reproducible examples of each implementation are made available through Google Colab.
