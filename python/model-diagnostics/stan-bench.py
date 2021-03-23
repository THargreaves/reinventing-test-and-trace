import os
import pickle

import numpy as np
import pandas as pd
import pystan
import scipy.special as sp
import scipy.stats as sts


# Parameters
P = [3, 5, 1, 4, 1, 2, 1, 1, 4]
N = 10 ** 3
ITER = 1000
RUNS = 10
WARMUP = 200
CHAINS = 8
SEED = 1729

# Setup environment
if not os.path.exists('samples'):
    os.mkdir('samples')

# GROUND TRUTH
np.random.seed(SEED)

# Mask-wearing
mask_wearer = np.random.binomial(1, 0.5, size=(N,))
true_mask_impact = np.random.lognormal(-1, 0.5, len(P))
true_mask_impact_rep = np.repeat(true_mask_impact, P)

# Test use
t_i = np.random.beta(8, 2, 1)  # Prob(tested | infected)
t_not_i = np.random.beta(2, 20, 1)  # Prob(tested | not-infected)
true_gamma = np.array([t_i, t_not_i])

# Test accuracy
test_sensitivity = np.random.beta(4, 3, 1)  # True positive rate
test_specificity = np.random.beta(50, 2, 1)  # True negative rate
true_lambda = np.array([test_sensitivity, test_specificity])

# Transmission rates
true_transmission_rate_mu = np.random.beta(2, 10, len(P))
true_transmission_rate_std = np.sqrt(sts.invgamma.rvs(a=100, size=len(P)))
true_transmission_rate = np.concatenate([
    sp.expit(sp.logit(mu) + np.random.normal(0, std, p))
    for p, mu, std
    in zip(P, true_transmission_rate_mu, true_transmission_rate_std)
])
base_rate = np.random.beta(2, 10, 1)

# Occurrence rates
true_occurrence_rate_mu = np.random.beta(2, 10, len(P))
true_occurrence_rate_std = np.sqrt(sts.invgamma.rvs(a=50, size=len(P)))
true_occurrence_rate = np.concatenate([
    sp.expit(sp.logit(mu) + np.random.normal(0, std, p))
    for p, mu, std
    in zip(P, true_occurrence_rate_mu, true_occurrence_rate_std)
])

# Set antigen test mean and std. error for TP and TN rates (for strong priors)

# True positive
mean_tp = 0.73000
se_tp = 0.04133

# True negative
mean_tn = 0.99680
se_tn = 0.00066

mean_rates = np.array([mean_tp, mean_tn])
se_rates = np.array([se_tp, se_tn])

# SIMULATE DATA
data = {}
for p in range(sum(P)):
    occurrence = np.random.binomial(1, true_occurrence_rate[p], N)
    # Impact of mask-wearing
    transmission_prob = sp.expit(sp.logit(true_transmission_rate[p]) +
                                 np.log(true_mask_impact_rep[p]) * mask_wearer)
    transmission = occurrence * np.random.binomial(1, transmission_prob)
    data[f'O{p+1}'] = occurrence
    data[f'T{p+1}'] = transmission

data['T0'] = np.random.binomial(1, base_rate, N)
X = pd.DataFrame(data)
z = X.loc[:, X.columns.str.startswith('T')].sum(axis=1)
y = (z > 0).astype(int)

# Resampling for test use
tested = y*np.random.binomial(1, true_gamma[0], N) + \
         (1-y)*np.random.binomial(1, true_gamma[1], N)
y = y[tested == 1]
X_survey = X[tested == 0].reset_index()
m_survey = mask_wearer[tested == 0]
X = X[tested == 1].reset_index()
m = mask_wearer[tested == 1]
X = X.loc[:, X.columns.str.startswith('O')]
X_survey = X_survey.loc[:, X_survey.columns.str.startswith('O')]
N = X.shape[0]
NA = X_survey.shape[0]

# Introduce false positives and negatives
y = y * np.random.binomial(1, true_lambda[0], N) + \
    (1-y) * np.random.binomial(1, (1-true_lambda[1]), N)

# Record classes
c = np.array([i + 1 for i, p in enumerate(P) for __ in range(p)])

# SAMPLING
# Compile model
sm = pystan.StanModel('full_mod.stan')

# Define model data
model_data = {
    'N': N, 'NA': NA, 'P': sum(P), 'K': len(P),
    'X': X.to_numpy(), 'y': y.to_numpy(), 'c': c, 'm': m, 'm_survey': m_survey,
    'survey': X_survey.to_numpy(), 'mean_lambda': mean_rates, 'se_lambda': se_rates,
}

# Fit model
for i in range(RUNS):
    # Check for existing run
    if os.path.exists(f'samples/samples_{i}.pkl'):
        continue

    fit = sm.sampling(data=model_data, iter=ITER, warmup=WARMUP, chains=CHAINS, verbose=True)

    with open(f'samples/samples_{i}.pkl', 'wb') as p:
        pickle.dump(fit, p)
