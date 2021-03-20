import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pystan
import arviz as arvz

class BaseModel:
    def __init__(self, model,N,P):
        self.model = gen
    def simulate_ground_truth(self, P):
        
        # Simulating transmission and occurence rates
        transmission_rate = np.random.beta(2, 10, P)
        occurrence_rate = np.random.beta(2, 10, P)
        base_rate = np.random.beta(2, 10, 1)
        
        self.ground_truth = {'true_occurence_rate':occurrence_rate, 
                            'true_theta':transmission_rate, 
                            'true_rho':base_rate}


        
    
