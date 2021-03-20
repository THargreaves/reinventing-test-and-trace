import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pystan
import arviz as arvz

from models import BaseModel
np.random.seed(SEED)

# Note: To comply with upcoming version of PyStan syntax (without model_name attribute) 
# Define Diagnotic object using Model object and str name

class Posetrior:
    """
    Posterior for base model
    
    :param N: fiting sample size
    :param P: number of settings
    """
    def __init__(self, model, N, P):
        self.model = model
        self.N = N
        self.P = P
        
    def simulate_ground_truth(self, P):
        
        # Simulating transmission and occurence rates
        transmission_rate = np.random.beta(2, 10, P)
        occurrence_rate = np.random.beta(2, 10, P)
        base_rate = np.random.beta(2, 10, 1)
        
        self.ground_truth = {'true_occurence_rate':occurrence_rate, 
                            'true_theta':transmission_rate, 
                            'true_rho':base_rate}
    
    def simulate_data(self):
        # Using
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
        

def runtime_lineplot(model_name, N_space):
    # Plots model runtime as function of fitting sample size
    if model_name == 'tt_base':
        
    
def runtime_lineplot(self, P_space):    
    # Plots runtime as function of theta vector dimension
    
def convergence_lineplot(self, N_space):
    # Plots model co as function of fitting sample size
    
def convergence_lineplot(self, P_space):
    # Plots runtime as a function of the number of thetas to estimate    
    
def mse_lineplot(self, P_space):
        # Plots runtime as a function of the number of thetas to estimate
        
        
    
