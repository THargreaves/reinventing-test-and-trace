import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pystan
import arviz as arvz
np.random.seed(SEED)

# Note: To comply with upcoming version of PyStan syntax (without model_name attribute) 
# Define Diagnotic object using Model object and str name

class Diagnostic:
    
    def __init__(self, model, name):
        self.model = model
        self.name = name
    
    def simulate_ground_truth(self, P):
        
        # Simulating transmission and occurence rates
        transmission_rate = np.random.beta(2, 10, P)
        occurrence_rate = np.random.beta(2, 10, P)
        base_rate = np.random.beta(2, 10, 1)
        
        # Simulating testing rates
        t_i = np.random.beta(8, 2, 1)  # Prob(tested | infected)
        t_not_i = np.random.beta(2, 20, 1)  # Prob(tested | not-infected)
        testing_rates = np.array([t_i, t_not_i])
        
        # Simulating test sensitivity and specificity rates
        test_sensitivity = np.random.beta(4, 3, 1)  # True positive rate
        test_specificity = np.random.beta(50, 2, 1)  # True negative rate
        test_accuracy_rates = np.array([test_sensitivity, test_specificity])
        return {'true_theta':transmission_rate, 'true_rho':base_rate, 'true_gamma':testing_rates, 'true_lambda':testing_rates}
        
    def simulate_data(self, N, P, ground_truth):
        # Simulating data
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
        
        if self.name == 'tt_base':
            X = X.loc[:, X.columns.str.startswith('O')]
            return {'N': N, 'P': P, 'X': X.to_numpy(), 'y': y.to_numpy()}
        
        elif self.name == 'tt_imperfect_testing':
            # Introducing false positives and negatives
            y = y*np.random.binomial(1, true_lambda[0], N) + (1-y)*np.random.binomial(1, (1-true_lambda[1]), N)
            X = X.loc[:, X.columns.str.startswith('O')]
            return {'N':N, 'P':P, 'X':X.to_numpy(), 'y':y.to_numpy(), 'mean_lambda':mean_rates, 'se_lambda':se_rates}

        elif self.name == 'tt_resampling': 
            # Resampling using testing probabilites conditional on infected
            tested = y*np.random.binomial(1, true_gamma[0], N) + (1-y)*np.random.binomial(1, true_gamma[1], N)
            y = y[tested == 1]
            X_survey = X[tested == 0].reset_index()
            X = X[tested == 1].reset_index()

            N = X.shape[0]
            NA = X_survey.shape[0]

            X = X.loc[:, X.columns.str.startswith('O')]
            X_survey = X_survey.loc[:, X_survey.columns.str.startswith('O')]
            return {'N':N, 'NA':NA, 'P':P, 'X':X.to_numpy(), 'y':y.to_numpy(), 'survey':X_survey.to_numpy()}
            
        elif self.name == 'tt_single_lvl_final':
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
            return {'N':N, 'NA':NA, 'P':P, 'X':X.to_numpy(), 'y':y.to_numpy(), 'survey':X_survey.to_numpy()}
        
    def runtime_lineplot(self, N_space):
        # Plots model runtime as function of fitting sample size
        
    def runtime_lineplot(self, P_space):    
        # Plots runtime as function of theta vector dimension
        
    def convergence_lineplot(self, N_space):
        # Plots model co as function of fitting sample size
        
    def convergence_lineplot(self, P_space):
        # Plots runtime as a function of the number of thetas to estimate    
        
    def mse_lineplot(self, P_space):
        # Plots runtime as a function of the number of thetas to estimate
        
        
    
