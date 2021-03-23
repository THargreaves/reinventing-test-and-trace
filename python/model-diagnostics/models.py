import time
import arviz
import numpy as np
import pandas as pd
import pystan

class BaseModel:
    def __init__ (self):
        self.model = pystan.StanModel('Desktop/GitHub/reinventing-test-and-trace-r/python/model-diagnostics/model-codes/base_model.stan')
        self.ground_truth = None
        self.data = None
        self.posterior = None
        self.runtime = None
        self.mse = None

    def simulate_data(self, N, P)->None:
        
        # Simulating transmission and occurence rates
        transmission_rate = np.random.beta(2, 10, P)
        occurrence_rate = np.random.beta(2, 10, P)
        base_rate = np.random.beta(2, 10, 1)
        
        data = {}
        for p in range(P):
            occurrence = np.random.binomial(1, occurrence_rate[p], N)
            transmission = occurrence * np.random.binomial(1, transmission_rate[p], N)
            data[f'O{p+1}'] = occurrence
            data[f'T{p+1}'] = transmission
        
        data['T0'] = np.random.binomial(1, base_rate, N)
        X = pd.DataFrame(data)
        z = X.loc[:, X.columns.str.startswith('T')].sum(axis=1)
        y = (z > 0).astype(int)
        X = X.loc[:, X.columns.str.startswith('O')]
        
        self.ground_truth = {'true_occurence_rate':occurrence_rate, 
                             'true_theta':transmission_rate, 
                             'true_rho':base_rate}
        
        self.data = {'N': N, 'P': P, 'X': X.to_numpy(), 'y': y.to_numpy()}
        
        
    def run(self, iterations, warmup_iterations, chains):
        print('Running model...')
        start = time.time()
        fit = self.model.sampling(data=self.data, iter=iterations, warmup=warmup_iterations, chains=chains)
        end = time.time()
        print('Finished running')
        error = np.sum((np.mean(fit.extract()['theta'], axis=0) - self.ground_truth['true_theta']) ** 2)
        self.mse = error
        self.posterior = fit
        self.runtime = (end - start) 
    
        