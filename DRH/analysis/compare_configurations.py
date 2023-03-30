'''
going "down" in resolution...
actually, there is an interesting question here: 
what is the difference between inferring based on n=4 and then going down to n=3
instead of directly inferring n=3. Is the landscape more/less spiky for n=4->n=3?
'''

import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt 
import re 
from fun import p_dist, bin_states
import seaborn as sns 

dir = '../data/sample_questions/mdl_config/'
param_files = [x for x in os.listdir(dir) if x.endswith('params.dat')]
param_files.sort()
param_files.insert(7, param_files.pop(0))

# basic states 
list_states = []
list_probabilities = []
for num, file in enumerate(param_files): 
    n = re.search(r"q(\d+)_nan5_q(\d+)", file).group(2)
    n = int(n)
    nJ = int(n*(n-1)/2)
    params = np.loadtxt(dir + file)
    J = params[:nJ]
    h = params[nJ:]

    probs =  p_dist(h, J)
    states = bin_states(n)
    
    if num == 0: 
        original_states = states
        list_states.append(states)
        list_probabilities.append(probs)
    
    else: 
        states_marginal = states[:, 0:3] 
        idx_marginal = np.array([np.where((states_marginal == i).all(1))[0] for i in original_states])
        prob_marginal = np.array([np.sum(probs[i]) for i in idx_marginal])
        list_states.append(states_marginal)
        list_probabilities.append(prob_marginal)

# relative changes 
for i in list_probabilities:
    plt.scatter(range(8), np.log(i), color = 'tab:blue', alpha = 0.5)
plt.xlabel('configurations')
plt.ylabel('log(probability)')


## new plot ## 

list_states = []
list_probabilities = []
for num, file in enumerate(param_files): 
    n = re.search(r"q(\d+)_nan5_q(\d+)", file).group(2)
    n = int(n)
    nJ = int(n*(n-1)/2)
    params = np.loadtxt(dir + file)
    J = params[:nJ]
    h = params[nJ:]

    probs = p_dist(h, J)
    list_probabilities.append(probs)
    
    #sns.histplot(x = [x/(2**(n-3)) for x in range(2**n)], weights = probs, discrete=True)

p1 = list_probabilities[0]
p2 = list_probabilities[1]
p3 = list_probabilities[2]
p4 = list_probabilities[3]
# a little bit closer 
# need the centering 
n=3
plt.bar(x = [x for x in range(2**n)], 
        height = p1, 
        align = 'center',
        alpha = 0.5,
        width=0.95)
n=4
plt.bar(x = [(x-0.5)/2 for x in range(2**n)], 
        height = p2, 
        align = 'center',
        alpha = 0.5,
        width=0.4)
n=5
plt.bar(x = [(x-1.5)/4 for x in range(2**n)],
        height = p3,
        align = 'center',
        alpha = 0.5,
        width = 0.17)
n=6 
plt.bar(x = [(x-3.5)/8 for x in range(2**n)],
        height = p4,
        align = 'center',
        alpha = 0.5,
        width = 0.08)

## do for n=3, n=6
p1 = list_probabilities[0]
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
p1_sort = np.sort(p1)
p1_norm = p1_sort/np.max(p1_sort)
plt.plot(np.log(p1_norm), color = 'tab:blue', alpha = 1)

# group into bins of 8
p4 = list_probabilities[3]
for i in range(8): 
    p4_i = p4[i*8:(i+1)*8]
    p4_i_sort = np.sort(p4_i)
    p4_i_norm = p4_i_sort/np.max(p4_i_sort)
    plt.plot(np.log(p4_i_norm), color = 'tab:orange', alpha = 0.5)   
plt.ylabel('log(probability)')
plt.xlabel('configurations (sorted)')
plt.suptitle('n=3 vs n=6')

## do for n=4, n=8
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
p4 = list_probabilities[1]
p8 = list_probabilities[5]
p4_norm = np.sort(p4)/np.max(p4)
plt.plot(np.log(p4_norm), color = 'tab:blue', alpha = 1)
for i in range(16): 
    p8_i = p8[i*16:(i+1)*16]
    p8_i_norm = np.sort(p8_i)/np.max(p8_i)
    plt.plot(np.log(p8_i_norm), color = 'tab:orange', alpha = 0.5)
plt.ylabel('log(probability)')
plt.xlabel('configurations (sorted)')
plt.suptitle('n=4 vs n=8')
