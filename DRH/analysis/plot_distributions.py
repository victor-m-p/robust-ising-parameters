from fun import p_dist 
import matplotlib.pyplot as plt
import os 
import numpy as np 
import re

fig, ax = plt.subplots()
dir = '../data/sample_questions/mdl_10/'
param_files = [x for x in os.listdir(dir) if x.endswith('params.dat')]
all_probabilities = []
for file in param_files: 
    n, i, j, samp = re.search(r"n(\d+)_i(\d+)_j(\d+)_sample(\d+)", file).group(1, 2, 3, 4)
    n, i, j, samp = int(n), int(i), int(j), int(samp)
    nJ = int(n*(n-1)/2)
    params = np.loadtxt(dir + file)
    J = params[:nJ]
    h = params[nJ:]

    probs =  p_dist(h, J)
    p_sort = np.sort(probs)[::-1]
    n_states = len(p_sort)
    
    plt.plot([i/(n_states-1) for i in range(n_states)], 
             [np.log(i*(n_states)) for i in p_sort], 
             alpha = 0.005, color = 'tab:orange')
    
    all_probabilities.append(p_sort)

all_probabilities = np.stack(all_probabilities)
plt.plot([i/(n_states-1) for i in range(n_states)],
         [np.log(i*(n_states)) for i in np.mean(all_probabilities, axis = 0)],
         color = 'tab:orange', lw = 3)

dir = '../data/sample_questions/mdl/'
param_files = [x for x in os.listdir(dir) if x.endswith('params.dat')]
all_probabilities = []
for file in param_files: 
    n, i, j, samp = re.search(r"n(\d)_i(\d+)_j(\d+)_sample(\d+)", file).group(1, 2, 3, 4)
    n, i, j, samp = int(n), int(i), int(j), int(samp)
    nJ = int(n*(n-1)/2)
    params = np.loadtxt(dir + file)
    J = params[:nJ]
    h = params[nJ:]

    probs =  p_dist(h, J)
    p_sort = sorted(probs)[::-1]
    n_states = len(p_sort)
    
    plt.plot([i/(n_states-1) for i in range(n_states)], 
             [np.log(i*(n_states)) for i in p_sort], 
             alpha = 0.005, color = 'tab:blue')
    
    all_probabilities.append(p_sort)
    
all_probabilities = np.stack(all_probabilities)
plt.plot([i/(n_states-1) for i in range(n_states)],
         [np.log(i*(n_states)) for i in np.mean(all_probabilities, axis = 0)],
         color = 'tab:blue', lw = 3)
plt.ylim(-15, 7)
plt.xlabel('rank/(total-1)')
plt.ylabel('log p*(total)')
plt.title('Probability of configurations')
plt.savefig('../figures/rank_vs_normp.pdf', bbox_inches = 'tight')
plt.savefig('../figures/rank_vs_normp.svg', bbox_inches = 'tight')