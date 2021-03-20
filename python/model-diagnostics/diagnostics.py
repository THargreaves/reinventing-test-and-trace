import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pystan
import arviz as arvz
np.random.seed(SEED)


class Diagnostic:
    
    def __init__(self, model):
        self.model = model    
    
    def simulate_ground_truth:
        
        # Simulating transmission and occurence rates
        true_transmission_rate = np.random.beta(2, 10, P)
        true_occurrence_rate = np.random.beta(2, 10, P)
        base_rate = np.random.beta(2, 10, 1)
        
        # Simulating testing rates
        t_i = np.random.beta(8, 2, 1)  # Prob(tested | infected)
        t_not_i = np.random.beta(2, 20, 1)  # Prob(tested | not-infected)
        true_gamma = np.array([t_i, t_not_i])
        
        # Simulating test sensitivity and specificity rates
        test_sensitivity = np.random.beta(4, 3, 1)  # True positive rate
        test_specificity = np.random.beta(50, 2, 1)  # True negative rate
        true_lambda = np.array([test_sensitivity, test_specificity])
        
    def simulate_data(self, N, P):
        # Simulate data
        data = {}
        for p in range(P):
            occurrence = np.random.binomial(1, true_occurrence_rate[p], N)
            transmission = occurrence * np.random.binomial(1, true_transmission_rate[p], N)
            data[f'O{p+1}'] = occurrence
            data[f'T{p+1}'] = transmission
        
        data['T0'] = np.random.binomial(1, base_rate, N)
        X = pd.DataFrame(data)
        z = X.loc[:, X.columns.str.startswith('T')].sum(axis=1)
        y = (z > 0).astype(int)
        
        if self.model.model_name == 'base_model':
            X = X.loc[:, X.columns.str.startswith('O')]
            return {'N': N, 'P': P, 'X': X.to_numpy(), 'y': y.to_numpy()}

        elif:
        # Resampling using testing probabilites conditional on infected
        tested = y*np.random.binomial(1, true_gamma[0], N) + (1-y)*np.random.binomial(1, true_gamma[1], N)
        y = y[tested == 1]
        X_survey = X[tested == 0].reset_index()
        X = X[tested == 1].reset_index()
        
        N = X.shape[0]
        NA = X_survey.shape[0]
        
        # Introducing false positives and negatives
        y = y*np.random.binomial(1, true_lambda[0], N) + (1-y)*np.random.binomial(1, (1-true_lambda[1]), N)
        
        X = X.loc[:, X.columns.str.startswith('O')]
        X_survey = X_survey.loc[:, X_survey.columns.str.startswith('O')]
        X.info()
        X_survey.info()
        

    def runtime_lineplot(self, N_list):
        # Plots model runtime for varying fitting data sample sizes
        
    def runtime_lineplot(self, P_list):    
        # Plots runtime for varying number of parameters
        
    def convergence_lineplot:
        # Plots runtime as a function of the number of thetas to estimate
        
    def convergence_lineplot:
        # Plots runtime as a function of the number of thetas to estimate    
        
    def mse_lineplot:
        # Plots runtime as a function of the number of thetas to estimate
        
        
    
