'''
VMP 2023-03-24: 
Running synthetic RCT.
Relies on functions from RCT_fun.py 
'''

import numpy as np
import pandas as pd 
from tqdm import tqdm 
import itertools 
import matplotlib.pyplot as plt 
from RCT_fun import *

# system size 
n_visible, n_hidden = 16, 0
n_nodes = n_visible + n_hidden

# setup 
intervention_var = 7 + n_hidden # ecstacy
outcome_var = 14 + n_hidden # major depression
enforce = True
procedure = 'Treatment' # Treatment & Prevention
n_timestep = 100
num_flips = 1

if procedure == 'Treatment': 
    outcome_start = 1
elif procedure == 'Prevention':
    outcome_start = -1
else: 
    raise ValueError('Procedure must be Treatment or Prevention')

# baseline configurations
questions = pd.read_csv(f'../data/preprocessing/questions_h{n_hidden}.csv')
probabilities = np.loadtxt(f'../data/preprocessing/prob_h{n_hidden}.txt')
configurations = np.loadtxt(f'../data/preprocessing/conf_h{n_hidden}.txt', dtype=int)

# get control population
idx_control = np.where((configurations[:, intervention_var] == -1) & (configurations[:, outcome_var] == outcome_start))[0]
p_control = np.copy(probabilities)
mask = np.ones_like(p_control, dtype=bool)
mask[idx_control] = False
p_control[mask] = 0
p_control = p_control / p_control.sum()

# get experiment population
### index does seem to be consistent so we can do this ###
idx_experiment = np.where((configurations[:, intervention_var] == 1) & (configurations[:, outcome_var] == outcome_start))[0]
nonzero_indices = np.nonzero(p_control)
p_nonzero = p_control[nonzero_indices]
max_idx = len(configurations)
p_experiment = np.zeros(max_idx)
p_experiment[idx_experiment] = p_nonzero

# general preparation
idx_outcome = np.where(configurations[:, outcome_var] == 1)[0]
evolve_arr = np.zeros((n_timestep, len(configurations)))
num_variables = len(configurations[0])
all_combinations = np.array(list(itertools.combinations(range(num_variables), num_flips)))
num_neighbors = len(all_combinations)
if enforce == True: 
    enforce_idx = intervention_var
else: 
    enforce_idx = num_variables + 100

# evolve experiment 
evolve_arr = np.zeros((n_timestep, len(configurations)))
p_current = p_experiment
for i in tqdm(range(n_timestep)): 
    p_evolve = push_forward(num_variables, 
                            configurations, 
                            probabilities,
                            p_current, 
                            all_combinations,
                            num_flips,
                            num_neighbors,
                            enforce_idx) # all numbers > num variables will lead to no enforcement
    evolve_arr[i, :] = p_evolve
    p_current = p_evolve
fraction_outcome = evolve_arr[:, idx_outcome].sum(axis=1)
np.savetxt(f'../data/RCT/experiment.{n_hidden}.{procedure}.{enforce}.{intervention_var}.{outcome_var}.{n_timestep}.{num_flips}.txt', fraction_outcome)

# evolve 
p_current = p_control
evolve_arr = np.zeros((n_timestep, len(configurations)))
for i in tqdm(range(n_timestep)): 
    p_evolve = push_forward(num_variables, 
                            configurations, 
                            probabilities,
                            p_current, 
                            all_combinations,
                            num_flips,
                            num_neighbors,
                            enforce_idx) # all numbers > num variables will lead to no enforcement
    evolve_arr[i, :] = p_evolve
    p_current = p_evolve
fraction_outcome = evolve_arr[:, idx_outcome].sum(axis=1)
np.savetxt(f'../data/RCT/control.{n_hidden}.{procedure}.{enforce}.{intervention_var}.{outcome_var}.{n_timestep}.{num_flips}.txt', fraction_outcome)

# if name equals main 
# maybe put into pandas format; but that could also be done later
# also much more that we want to learn of course 
# could push a lot of this to tomorrow 
# but pretty exciting stuff 