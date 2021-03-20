import time
import arviz
import numpy as np
import pandas as pd
import stan

class BaseModel:
    def __init__ (self, N, P):
        self.N = N
        self.P = P 

    def simulate_ground_truth(self):
        
        # Simulating transmission and occurence rates
        transmission_rate = np.random.beta(2, 10, self.P)
        occurrence_rate = np.random.beta(2, 10, self.P)
        base_rate = np.random.beta(2, 10, 1)
        
        self.ground_truth = {'true_occurence_rate':occurrence_rate, 
                            'true_theta':transmission_rate, 
                            'true_rho':base_rate}
    def simulate_data(self):
        # Simulating data
        data = {}
        for p in range(self.P):
            occurrence = np.random.binomial(1, true_occurrence_rate[p], N)
            transmission = occurrence * np.random.binomial(1, true_transmission_rate[p], N)
            data[f'O{p+1}'] = occurrence
            data[f'T{p+1}'] = transmission
        
        data['T0'] = np.random.binomial(1, base_rate, N)
        X = pd.DataFrame(data)
        z = X.loc[:, X.columns.str.startswith('T')].sum(axis=1)
        y = (z > 0).astype(int)
        X = X.loc[:, X.columns.str.startswith('O')]
        return {'N': N, 'P': P, 'X': X.to_numpy(), 'y': y.to_numpy()}