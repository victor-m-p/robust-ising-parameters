'''
VMP 2023-03-24: 
Prepares key documents for the analysis of NSDUH data.
'''

import numpy as np 
from fun import p_dist, bin_states
import pandas as pd 
pd.set_option('display.max_colwidth', None)

## NB: this will change with hidden nodes
## we need a system for handling this 
## and we need to know what the order is 
## For hidden nodes (information is somewhere).
n_visible, n_hidden = 16, 0
n_nodes = n_visible + n_hidden

d = pd.read_csv('../data/reference/NSDUH_2008_2019_NAN5_SUIC.csv')
questions_original = list(d.columns)

## question short-hands
questions_readable = [
    'Cigarette', # by year
    'Alcohol', # by year
    'Marijuana', # by year
    'Cocaine', # by year
    'Heroin', # by year
    'LSD', # by year
    'PCP', # by year
    'Ecstasy', # by year
    'Inhalants', # by year
    'Pain relievers', # by year
    'Tranquilizers', # by year
    'Stimulants', # by year
    'Sedatives', # by year
    'Suicide thoughts', # by year
    'Major depression', # by year
    'Psychological distress' # by year
]

question_reference = pd.DataFrame({
    'question': [f'hidden{x+1}' for x in range(n_hidden)] + questions_readable,
    'question_NSDUH': [f'hidden{x+1}' for x in range(n_hidden)] + questions_original
    })

# question_id is going to be tricky because 
# it will not be consistent across
# will have to merge on "question" in any case ...
question_reference['question_id'] = question_reference.index + 1
question_reference.to_csv(f'../data/preprocessing/questions_h{n_hidden}.csv', index = False)

# calculate probability of all configurations based on parameters h, J.
params = np.loadtxt(f'../data/mdl/hidden_{n_hidden}.txt', dtype=float, delimiter=',')
nJ = int(n_nodes*(n_nodes-1)/2)
J = params[:nJ]
h = params[nJ:]
p = p_dist(h, J) # takes a minute (and a lot of memory). 
np.savetxt(f'../data/preprocessing/prob_h{n_hidden}.txt', p)

# all configurations file allstates 
allstates = bin_states(n_nodes) # takes a minute (do not attempt with n_nodes > 20)
np.savetxt(f'../data/preprocessing/conf_h{n_hidden}.txt', allstates.astype(int), fmt='%i')