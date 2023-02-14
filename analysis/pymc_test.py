import numpy as np 
import pymc as pm 
import matplotlib.pyplot as plt 
import pandas as pd 
import arviz as az 

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0] * X1 + beta[1] * X2 + rng.normal(size=size) * sigma

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
axes[0].scatter(X1, Y, alpha=0.6)
axes[1].scatter(X2, Y, alpha=0.6)
axes[0].set_ylabel("Y")
axes[0].set_xlabel("X1")
axes[1].set_xlabel("X2");

basic_model = pm.Model() # creates a model object (container for random variables)
with basic_model: # creates context manager 
    # Priors for unknown model parameters (stochastic)
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Expected value of outcome
    mu = alpha + beta[0] * X1 + beta[1] * X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

with basic_model:
    # draw 1000 posterior samples
    idata = pm.sample()
    
az.plot_trace(idata, combined=True);
az.summary(idata, round_to=2)


####### our data #########
import numpy as np
from fun import p_dist, bin_states, fast_logsumexp

# setup
n = 3
C = 5000
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

# we will eventually do this smarter ...
X1 = sample[:, 0]
X2 = sample[:, 1]
Y = sample[:, 2]
X1[X1 < 0] = 0
X2[X2 < 0] = 0
Y[Y < 0] = 0

# There should be some "p" and then Bernoulli...
test_modelx = pm.Model() 
with test_modelx: 
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta", mu=0, sigma=1, shape=2) # shape ...
    #interaction = pm.Normal("interaction", mu=0, sigma=1)
    p = pm.Deterministic("p", pm.math.invlogit(alpha + beta[0]*X1 + beta[1]*X2))
    #p = pm.math.invlogit(alpha + beta[0]*X1 + beta[1]*X2)
    outcome = pm.Bernoulli("outcome", p, observed=Y)

with test_modelx:
    idata = pm.sample()

az.plot_trace(idata, var_names=['alpha', 'beta'], combined=True);
az.summary(idata, var_names=['alpha', 'beta'], round_to=2)
az.summary(idata, round_to=2) 
beta0 = 3.49 # change in log odds I think
beta1 = -1.71 # change in log odds I think 
invlogit = lambda x: 1 / (1 + np.exp(-x))
invlogit(3.49 - 1.71) # probability of 1 if 1/1 # (85.5%)
invlogit(3.49 + 1.71) # probability of 1 if 1/0 # (99.5%)
invlogit(-3.49 - 1.71) # probability of 1 if 0/0 (0.5%)
invlogit(-3.49 + 1.71) # probability of 1 if 0/1 (14.4%)

# what does the sample say?
h # it does (very slightly) capture that alpha is positive 
J # so it does make the correct inference that 
# * J_{13} positive
# * J_{23} negative
# * J_{12} around double influence 
invlogit(-1) # 0.27 

##### try another one #####
test_modely = pm.Model()
with test_modely: 
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta", mu=0, sigma=1, shape=2)
    p = pm.Deterministic("p", pm.math.invlogit(alpha + beta[0]*X1 + beta[1]*Y))
    outcome = pm.Bernoulli("outcome", p, observed=X2)

with test_modely:
    idata = pm.sample()

az.plot_trace(idata, var_names = ['alpha', 'beta'], combined=True);
az.summary(idata, var_names=['alpha', 'beta'], round_to=2)
az.summary(idata, var_names=['alpha', 'beta', 'p'], round_to=2)

h # gets alpha (mean) wrong it seems 
J # J_{12} is (correctly) positive (and the strongest in the system), J_{23} is correctly negative (and the same)
alpha = -1.03
beta0 = 4.38
beta1 = -1.71

#### some calculus here that I need to understand #### 
allstates = bin_states(n, True)  # all 2^n possible binary states in {-1,1} basis

not_one_not_two = np.mean(probabilities[0:2])
not_one_two = np.mean(probabilities[2:4])
one_not_two = np.mean(probabilities[4:6])
one_two = np.mean(probabilities[6:8])

one_two/one_not_two