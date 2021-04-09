import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import stan
import arviz as arvz

from models import BaseModel
np.random.seed(23)




def runtime_lineplot_N(model, N_space, P=4):
    """
    Plots model runtime as a function of the number of fitting samples
    Saves plot as .png file
    
    :param model: A Test and Trace model object
    :param N_space: One-dimensional iterable of integers for each number of fitting samples
    
    """ 
    N_space = np.sort(np.array(N_space))
    runtimes = []
    for i in N_space:
        N = int(i)
        model.simulate_data(N,P)
        model.run(iterations=1000, warmup_iterations=500, chains=4)
        runtimes.append(model.runtime)
    
    fig, ax = plt.subplots()
    ax.plot(N_space, runtimes, color='tab:orange')
    ax.set_ylabel('Runtime')
    ax.set_xlabel('Number of fitting samples')
    ax.set_title('Model Runtime')
    fig.tight_layout()
    plt.savefig('runtime_N_plot.png')
    plt.show()
    
    
    
        
    
def runtime_lineplot(model, P_space, N = 5000):    
    """
    Plots model runtime as a function its number of theta parameters
    
    :param model:
    :param P_space: Iterable of integers for each number of theta parameters
    
    """ 
    P_space = np.sort(np.array(P_space))
    runtimes = []
    for i in P_space:
        model.simulate_data(N,i)
        model.run()
        runtimes.append(model.runtime)
    
    fig, ax = plt.subplots()
    ax.plot(P_space, runtimes, color='tab:blue')
    ax.set_ylabel('Runtime')
    ax.set_xlabel('Number of theta parameters')
    ax.set_title('Model Runtime')
    fig.tight_layout()
    plt.savefig('runtime_P_plot.png')
    plt.show()
    
    
    
    
    
    
def convergence_lineplot(model, N_space):
    pass
    
    
    
    
def convergence_lineplot(model, P_space):
    # Plots runtime as a function of the number of thetas to estimate
    pass
    
    
    
def mse_lineplot(model, N_space, P=4):
    """
    Plots mean squared error of model estimates as a function of the number of fitting samples
    
    :param model:
    :param P_space: Iterable of integers for each number of theta parameters
    
    """ 
    N_space = np.sort(np.array(N_space))
    runtimes = []
    for i in N_space:
        model.simulate_data(i,P)
        model.run()
        runtimes.append(model.runtime)
    
    fig, ax = plt.subplots()
    ax.plot(N_space, runtimes, color='tab:red')
    ax.set_ylabel('Runtime')
    ax.set_xlabel('Number of fitting samples')
    ax.set_title('Model MSE')
    fig.tight_layout()
    plt.savefig('mse_N_plot.png')
    plt.show()
    
    
def mse_lineplot(model, P_space):
    pass
    
def mape_lineplot(model, N_space):
    pass
        
        
    
