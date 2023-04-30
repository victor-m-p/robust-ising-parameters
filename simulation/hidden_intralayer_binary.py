import numpy as np 
import itertools 

# taken from coniii enumerate
def fast_logsumexp(X, coeffs=None):
    """correlation calculation in Ising equation

    Args:
        X (np.Array): terms inside logs
        coeffs (np.Array, optional): factors in front of exponential. Defaults to None.

    Returns:
        float: sum of exponentials
    """
    Xmx = max(X)
    if coeffs is None:
        y = np.exp(X-Xmx).sum()
    else:
        y = np.exp(X-Xmx).dot(coeffs)

    if y<0:
        return np.log(np.abs(y))+Xmx, -1.
    return np.log(y)+Xmx, 1.

# still create J_combinations is slow for large number of nodes
def p_dist(h, J):
    """return probabilities for 2**h states

    Args:
        h (np.Array): local fields
        J (np.Array): pairwise couplings. 

    Returns:
        np.Array: probabilities for all configurations
    """
    n_nodes = len(h)
    hJ = np.concatenate((h, J))
    h_combinations = np.array(list(itertools.product([1, -1], repeat = n_nodes)))
    J_combinations = np.array([list(itertools.combinations(i, 2)) for i in h_combinations])
    J_combinations = np.add.reduce(J_combinations, 2)
    J_combinations[J_combinations != 0] = 1
    J_combinations[J_combinations == 0] = -1
    condition_arr = np.concatenate((h_combinations, J_combinations), axis = 1)
    flipped_arr = hJ * condition_arr
    summed_arr = np.sum(flipped_arr, axis = 1)
    logsumexp_arr = fast_logsumexp(summed_arr)[0]
    Pout = np.exp(summed_arr - logsumexp_arr)
    return Pout[::-1]

np.exp(1.2) + np.exp(-1.4) + np.exp(-0.2) + np.exp(0.4)
np.exp(0.4)

1.5/5.877


# try extremely simple system
h = np.array([-0.1, 0.5])
J = np.array([0.8])
hJ = np.concatenate((h, J))
hJ
np.exp(-0.8)
h_combinations = np.array(list(itertools.product([1, -1], repeat = 2)))
h_combinations
J_combinations = np.array([list(itertools.combinations(i, 2)) for i in h_combinations])
J_combinations = np.add.reduce(J_combinations, 2)
J_combinations[J_combinations != 0] = 1
J_combinations[J_combinations == 0] = -1
condition_arr = np.concatenate((h_combinations, J_combinations), axis = 1)
flipped_arr = hJ * condition_arr # 
summed_arr = np.sum(flipped_arr, axis = 1)
logsumexp_arr = fast_logsumexp(summed_arr)[0] # not an 
Pout = np.exp(summed_arr - logsumexp_arr)
summed_arr
logsumexp_arr
np.exp(0.4-1.77)
Pout[::-1] # probabilities for each combination 
Pout[::-1]
h_combinations
np.exp(1.2-1.77)
import scipy.stats as stats 
percentile = 0.7
z_score = stats.norm.ppf(percentile, scale=0.5) # std important 
samples = np.random.normal(loc=z_score, scale=0.5, size=1000) # std important
len(samples[samples>0])/1000

# logsumexp: 
X = summed_arr
Xmx = max(X)
coeffs = None 
if coeffs is None:
    y = np.exp(X-Xmx).sum()
else:
    y = np.exp(X-Xmx).dot(coeffs)

if y<0:
    return np.log(np.abs(y))+Xmx, -1.
return np.log(y)+Xmx, 1.

np.log(y)+Xmx












## alternative 1: 
## does not quite work because we manually specify covariance matrix ...
## what we would like is to specify means of samples and correlation. 
num_samples = 400

# The desired mean values of the sample.
mu = np.array([5.0, 0.0, 10.0])

# The desired covariance matrix.
r = np.array([
        [  3.40, -2.75, -2.00],
        [ -2.75,  5.50,  1.50],
        [ -2.00,  1.50,  1.25]
    ])

# Generate the random samples.
rng = np.random.default_rng()
y = rng.multivariate_normal(mu, r, size=num_samples)

## alternative 2: 
# (1) sample the first param 
# (2) use this to sample the second param ...
# suggests a causal structure rather than just correlational 