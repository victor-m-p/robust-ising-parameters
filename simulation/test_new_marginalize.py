import numpy as np 
from sample_functions import marginalize_n 

def marginalize_nn(configurations: np.array, probabilities: np.array, n_hidden: int):
    '''
    marginalize the hidden states and return the marginalized 
    unique configurations and probabilities.
    it is assumed that the hidden nodes are the first n columns.
    '''
    # Remove the first n columns (elements) from configurations
    reduced_configurations = configurations[:, n_hidden:]

    # Find the unique configurations in reduced_configurations and their inverse
    unique_configurations, inverse = np.unique(reduced_configurations, axis=0, return_inverse=True)

    # Sum the probabilities corresponding to the same configuration using bincount
    marginalized_probs = np.bincount(inverse, weights=probabilities)

    return unique_configurations, marginalized_probs

p = np.array([0.1, 0.2, 0.3, 0.4])
configs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
n_hidden = 1
x = marginalize_n(configs, p, n_hidden)
xx = marginalize_nn(configs, p, n_hidden)

x
xx

reduced_configs = configs[:, n_hidden:]
uniq, inv = np.unique(reduced_configs, axis=0, return_inverse=True)
inv
uniq

