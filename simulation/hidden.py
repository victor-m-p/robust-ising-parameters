import numpy as np 
import pandas as pd 

# 1.
# h distributed as Normal(0, 1)
# w distributed as Normal(0, 1)
# h_samp distributed as Bernoulli(logit(h))
# v_samp distributed as Bernoulli(logit(h_samp*w))

# the simple model
n_hidden = 2 # number of hidden units
n_visible = 5 # number of visible units

# sample from random distribution 
hidden_bias = np.random.normal(0, 1, n_hidden) # bias of hidden layer

# connect to the visible layer
weight = np.random.normal(0, 1, (n_visible, n_hidden)) # weight from hidden to visible layer

# sample 100 observations of visible layer based on hidden_bias and weight
n_sample = 10000
samp_visible = np.zeros((n_sample, n_visible))
for i in range(n_sample):
    samp_hidden = np.random.binomial(1, 1/(1+np.exp(-hidden_bias))) # sample hidden
    samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(samp_hidden, weight.T))))) # sample visible

# basic statistics 
np.mean(samp_visible, axis=0) # we do get non-equal weighting 
np.corrcoef(samp_visible, rowvar=False) # we do not get strong correlations 

'''
no strong correlations between the visible units, but the weights differ some. 
'''

# 2.
# h distributed as Normal(0, 1)
# w distributed as Normal(0, 1)
# v_samp distributed as Bernoulli(logit(h*w))
samp_visible = np.zeros((n_sample, n_visible))
for i in range(n_sample):
    samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(hidden_bias, weight.T)))))

# basic statistics
np.mean(samp_visible, axis=0) # we do get non-equal weighting 
np.corrcoef(samp_visible, rowvar=False) # tends to 0 

'''
no strong correlations between the visible units, but the weights differ some.
the correlations are very different from the previous case (same in limit?) 
correlations in the visible layer appear to tend to 0; 
'''

# 3. 
# the first model again, but each person different h 
# we assume that w is the same across people 
samp_visible = np.zeros((n_sample, n_visible))
for i in range(n_sample): 
    hidden_bias = np.random.normal(0, 1, n_hidden) # bias of hidden layer
    samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(hidden_bias, weight.T)))))

np.mean(samp_visible, axis=0) 
np.corrcoef(samp_visible, rowvar=False) 

'''
I still do not quite get what I was hoping for ...
'''

# 4. 
# try with just 1 hidden unit and 3 outcomes (IQ tests, say). 
# here all the same weights (positive) so positively correlated. 
# just gets much harder to mentally decompose in previous cases. 
n_visible = 3
n_hidden = 1
samp_visible = np.zeros((n_sample, n_visible))
weight = np.array([[5, 5, 5]]).T # weight from hidden to visible layer
for i in range(n_sample): 
    hidden_bias = np.random.normal(0, 1, n_hidden) # bias of hidden layer
    samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(hidden_bias, weight.T)))))

np.mean(samp_visible, axis=0) # yes; should give 0.5 because equally likely to sample "smart" and "dumb"
np.corrcoef(samp_visible, rowvar=False) # here is a correlation; design correct but needs large values.

'''
We can get correlation if we really want to.
Here we get means towards 0 because hidden_bias sampled from a normal centered on 0. 
'''

# 5.
# adding error to the model ?
n_visible = 3
n_hidden = 1
samp_visible = np.zeros((n_sample, n_visible))
weight = np.array([[5, 5, 5]]).T # weight from hidden to visible layer
for i in range(n_sample): 
    error = np.random.normal(0, 1, n_visible) # error term
    error = error.reshape((3, 1))
    weight_error = weight + error
    hidden_bias = np.random.normal(0, 1, n_hidden) # bias of hidden layer
    samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(hidden_bias, weight_error.T)))))

np.mean(samp_visible, axis=0) # yes; should give 0.5 because equally likely to sample "smart" and "dumb"
np.corrcoef(samp_visible, rowvar=False)

'''
I think this makes sense.
We should try to formalize this 
'''

# 6. 
# going to more than one level of hidden nodes 

## number of units
n_hidden_2 = 1 # n deep layer
n_hidden_1 = 2 # n shallow layer
n_visible = 5 # n visible

## population for deepest layer
# ...

## connect them
weight_hh = np.random.normal(...) # weight hidden to hidden
weight_hv = np.random.normal(...) # weight hidden to visible

## sample 

'''
Things we want to be able to infer: 
(1) Given knowledge of N hidden; what are the connections (maybe also distributions "mean" for hidden).
(2) Given we do not know; also estimate best N (and the properties).
'''

''' 
Simulation challenges: 
(1) connections within layers
(2) bi-directional connections between layers?
'''