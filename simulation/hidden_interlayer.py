import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# 1. first model 
## main code 
n_sample = 10000 # all independent (no repeated measurement)
n_visible = 3 # number of visible units
n_hidden = 1 # number of hidden units 
weight = np.random.normal(0, 1, (n_visible, n_hidden)) # weight from hidden to visible layer
samp_visible = np.zeros((n_sample, n_visible)) # init samples of visible units
for i in range(n_sample): 
    error = np.random.normal(0, 1, n_visible).reshape((n_visible,n_hidden)) # error term
    weight_error = weight + error # error on weight 
    hidden_bias = np.random.normal(0, 1, n_hidden) # bias of hidden layer
    samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(hidden_bias, weight_error.T))))) # yi

np.mean(samp_visible, axis=0) 
np.corrcoef(samp_visible, rowvar=False) 
np.cov(samp_visible.T) # uncorrelated 

## test code (think hidden = "g" and visible = 3 IQ tests taken by 1000 people)
n_sample = 1000
n_visible = 3
n_hidden = 1
weight = np.array([[5, 5, 5]]).T
samp_visible = np.zeros((n_sample, n_visible))
for i in range(n_sample): 
    error = np.random.normal(0, 1, n_visible).reshape((n_visible, n_hidden))
    weight_error = weight + error # error on weight 
    hidden_bias = np.random.normal(0.5, 1, n_hidden) # bias of hidden layer
    samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(hidden_bias, weight_error.T))))) # yi

np.mean(samp_visible, axis=0) # > 0.5 because 0.5 in hidden bias. 
np.corrcoef(samp_visible, rowvar=False) # strongly positively correlated 

# 2. second model (sample hidden)
n_sample = 10000 
n_visible = 3 
n_hidden = 1 
weight = np.random.normal(0, 1, (n_visible, n_hidden)) 
hidden_bias = np.random.normal(0, 1, n_hidden) 
samp_visible = np.zeros((n_sample, n_visible)) 
for i in range(n_sample): 
    error = np.random.normal(0, 1, n_visible).reshape((n_visible,n_hidden)) 
    weight_error = weight + error 
    hidden_bias = np.random.normal(0, 1, n_hidden) 
    samp_hidden = np.random.binomial(1, 1/(1+np.exp(-hidden_bias))) # only difference 
    samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(hidden_bias, weight_error.T))))) 

np.mean(samp_visible, axis=0) 
np.corrcoef(samp_visible, rowvar=False) # is correlation 
np.cov(samp_visible.T) # no covariance

## test code (think hidden = "g" and visible = 3 IQ tests taken by 1000 people)
## why is this less correlated than the above?
n_sample = 1000
n_visible = 3
n_hidden = 1
weight = np.array([[5, 5, 5]]).T
samp_visible = np.zeros((n_sample, n_visible))
for i in range(n_sample): 
    error = np.random.normal(0, 1, n_visible).reshape((n_visible, n_hidden))
    weight_error = weight + error # error on weight 
    hidden_bias = np.random.normal(0.5, 1, n_hidden) # bias of hidden layer
    samp_hidden = np.random.binomial(1, 1/(1+np.exp(-hidden_bias))) # only difference 
    samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(samp_hidden, weight_error.T))))) # yi

np.mean(samp_visible, axis=0) # higer mean than previous 
np.corrcoef(samp_visible, rowvar=False) # lower coupling than previous 

# 3. going to more layers 
n_samples = 1000
## number of units
n_hidden_2 = 1 # n deep layer
n_hidden_1 = 2 # n shallow layer
n_visible = 3 # n visible

## connect them
weight_hh = np.random.normal(0, 1, (n_hidden_1, n_hidden_2)) # weight hidden to hidden
weight_hv = np.random.normal(0, 1, (n_visible, n_hidden_1)) # weight hidden to visible

## sample 
samp_visible = np.zeros((n_sample, n_visible))
for i in range(n_sample): 
    hidden_bias = np.random.normal(0, 1, n_hidden_2) # hidden bias for deepest layer
    logit_hidden_1 = 1/(1+np.exp(-(np.dot(hidden_bias, weight_hh.T)))) # sample shallow layer
    samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(logit_hidden_1, weight_hv.T))))) # sample visible layer

np.mean(samp_visible, axis=0) 
np.corrcoef(samp_visible, rowvar=False) 

## test this 
n_hidden_2 = 1 # n deep layer
n_hidden_1 = 2 # n shallow layer
n_visible = 3 # n visible

## connect them
weight_hh = np.array([[5, -5]]).T # weight hidden to hidden
weight_hv = np.array([[5, 5, 5],
                      [-5, -5, -5]]).T # weight hidden to visible

## sample 
samp_visible = np.zeros((n_sample, n_visible))
for i in range(n_sample): 
    hidden_bias = np.random.normal(0, 1, n_hidden_2) # hidden bias for deepest layer
    logit_hidden_1 = 1/(1+np.exp(-(np.dot(hidden_bias, weight_hh.T)))) # sample shallow layer
    samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(logit_hidden_1, weight_hv.T))))) # sample visible layer

np.mean(samp_visible, axis=0) 
np.corrcoef(samp_visible, rowvar=False) # strongly coupled together (reasonable)

# making something general
n_nodes = [1, 2, 3]
weights = [np.random.normal(0, 1, 2),  
           np.random.normal(0, 1, 3)]
bias = np.random.normal(0, 1, 1)

def sample_mania(node_list: list, 
                 weight_list: list,
                 bias,
                 n_sample: int): 
    
    n_visible = node_list[-1] # number visible layer 
    samp_visible = np.zeros((n_sample, n_visible)) # initialize visible layer
    
    w_hh = weight_list[:-1] # hidden-to-hidden
    w_hv = weight_list[-1] # hidden-to-visible
    
    for i in range(n_sample): 
        hidden_bias = bias # does bias only apply to lowest level 
        for w in w_hh: # loop through levels 
            logit_hidden = 1/(1+np.exp(-(np.dot(hidden_bias, w.T))))
            
        samp_visible[i] = np.random.binomial(1, 1/(1+np.exp(-(np.dot(logit_hidden, w_hv.T))))) # sample visible layer
    

## questions
# (1) where do we want to include error in this model?
# (2) how do we want to sample the hidden nodes? 


# general thoughts

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

'''
Potentials: 
(1) do we want to "knock out" some connections and try to infer this?
'''