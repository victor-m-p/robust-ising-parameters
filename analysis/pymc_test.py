import numpy as np 
import pymc as pm 
import matplotlib.pyplot as plt 
import pandas as pd 
import arviz as az 
from fun import p_dist, bin_states, fast_logsumexp

# setup
n = 3
C = 5000
np.random.seed(0)
invlogit = lambda x: 1 / (1 + np.exp(-x))

# generate params and probabilities
h = np.random.normal(scale=0.5, size=n)
J = np.random.normal(scale=0.5, size=n*(n-1)//2)
hJ = np.concatenate((h, J))
probabilities = p_dist(h, J) # potential culprit 

# visualize data 
allstates = bin_states(n, True)  # all 2^n possible binary states in {-1,1} basis
sample = allstates[np.random.choice(range(2**n), # doesn't have to be a range
                                    size=C, # how many samples
                                    replace=True, # a value can be selected multiple times
                                    p=probabilities)]  # random sample from p(s)

'''
# what are actually the correlations?
means = np.mean(sample, axis=0)
correlations = np.corrcoef(sample, rowvar=False)
correlations
corr_list = []
for i, _ in enumerate(correlations): 
    for j, _ in enumerate(correlations): 
        if i > j: 
            corr_list.append(correlations[i,j])
correlations = np.array(corr_list)
correlations
'''

# setup
n_samples = 10000
n_chains = 4
n_beta = 2

# setup
X = np.copy(sample)
X[X < 0] = 0

# 1 ~ 2 + 3
# 2 ~ 1 + 3
# 3 ~ 1 + 2
idata_list = []
combination_list = []
combinations = [(1, 2, 0), (0, 2, 1), (0, 1, 2)]
for i, j, k in combinations: 
    model = pm.Model()
    with model: 
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=2)
        p = pm.Deterministic("p", pm.math.invlogit(alpha + beta[0]*X[:, i] + beta[1]*X[:, j]))
        outcome = pm.Bernoulli("outcome", p, observed=X[:, k])
    with model: 
        idata = pm.sample()
    idata_list.append(idata)
    combination_list.append([i, j, k])

idata_h1 = idata_list[0]
idata_h2 = idata_list[1]
idata_h3 = idata_list[2]


idata_h1.posterior['beta'].values.reshape(4000, 2)

-2*1.12 - 2*0.93 + 2*.88 # 

0.933779*4

h
J
az.summary(idata_h1, var_names=['alpha', 'beta'], round_to=2)
az.summary(idata_h2, var_names=['alpha', 'beta'], round_to=2)
az.summary(idata_h3, var_names=['alpha', 'beta'], round_to=2)

## seems unreasonalbe ##
# h1 = 0.88 --> a1 = -1.93
# h2 = 0.2 --> a2 = -1.04
# h3 = 0.489 --> a3 = 0.02 

## seems pretty reasonable ## 
# J12 = 1.12 -> (4.3, 4.39) -- matches (pretty) well with 4*Jij
# J13 = 0.93 -> (3.36, 3.49) -- matches (pretty) well with 4*jij
# J23 = -0.49 -> (-1.7, -1.71) -- matches (pretty) well with 4*jij

np.mean(X[:, 0]) # 0.925
np.mean(X[:, 1]) # 0.803
np.mean(X[:, 2]) # 0.8426 

invlogit(0.02) # 0.5 
invlogit(-1) # 0.27 
invlogit(-1.93) # 0.13 (does not make sense it seems)


'''
## without intercept ## 
idata_list_noint = []
combination_list_noint = []
combinations = [(1, 2, 0), (0, 2, 1), (0, 1, 2)]
for i, j, k in combinations: 
    model = pm.Model()
    with model: 
        beta = pm.Normal("beta", mu=0, sigma=1, shape=2)
        p = pm.Deterministic("p", pm.math.invlogit(beta[0]*X[:, i] + beta[1]*X[:, j]))
        outcome = pm.Bernoulli("outcome", p, observed=X[:, k])
    with model: 
        idata = pm.sample()
    idata_list_noint.append(idata)
    combination_list_noint.append([i, j, k])

idata_h1_noint = idata_list_noint[0]
idata_h2_noint = idata_list_noint[1]
idata_h3_noint = idata_list_noint[2]

az.summary(idata_h1_noint, var_names=['beta'], round_to=2)
az.summary(idata_h2_noint, var_names=['beta'], round_to=2)
az.summary(idata_h3_noint, var_names=['beta'], round_to=2)
'''


# try for larger system 
n = 5
C = 20000
np.random.seed(0)

# generate params and probabilities
h = np.random.normal(scale=0.5, size=n)
J = np.random.normal(scale=0.5, size=n*(n-1)//2)
hJ = np.concatenate((h, J))
probabilities = p_dist(h, J)

# visualize data 
allstates = bin_states(n, True)  # all 2^n possible binary states in {-1,1} basis
sample = allstates[np.random.choice(range(2**n), # doesn't have to be a range
                                    size=C, # how many samples
                                    replace=True, # a value can be selected multiple times
                                    p=probabilities)]  # random sample from p(s)
X = np.copy(sample)
X[X < 0] = 0 

model = pm.Model()
with model: 
    alpha = pm.Normal("alpha", mu=0, sigma=5)
    beta = pm.Normal("beta", mu=0, sigma=5, shape=4)
    p = pm.Deterministic("p", pm.math.invlogit(alpha + beta[0]*X[:, 0] + beta[1]*X[:, 1] + beta[2]*X[:, 2] + beta[3]*X[:, 3]))
    outcome = pm.Bernoulli("outcome", p, observed=X[:, 4])
with model: 
    idata = pm.sample()

az.summary(idata, var_names=['alpha', 'beta'], round_to=2)

import itertools
combinations = list(itertools.combinations([1, 2, 3, 4, 5], 2))
one_five = J[3] # -0.0516
two_five = J[6] # 0.727
three_five = J[8] # 0.0608
four_five = J[9] # 0.222

# Jij 
one_five*4 # within CI
two_five*4 # within CI
three_five*4 # within CI
four_five*4 # within CI 


# 0.93 is the hi 
# -0.27 is the alpha 

# loop through and save the params # 
samples_alpha = idata.posterior['alpha'].values
samples_beta = idata.posterior['beta'].values.reshape(n_samples*n_chains, n_beta)
beta1 = samples_beta[:, 0].shape
beta2 = samples_beta[:, 1].shape


# ...
beta1_mean = np.mean(beta1)
beta1_median = np.median(beta1)
beta2_mean = np.mean(beta2)
beta2_median = np.median(beta2)
alpha_median = np.median(samples_alpha)
alpha_mean = np.mean(samples_alpha)

#### some calculus here that I need to understand #### 
allstates = bin_states(n, True)  # all 2^n possible binary states in {-1,1} basis

not_one_not_two = np.mean(probabilities[0:2])
not_one_two = np.mean(probabilities[2:4])
one_not_two = np.mean(probabilities[4:6])
one_two = np.mean(probabilities[6:8])

one_two/one_not_two