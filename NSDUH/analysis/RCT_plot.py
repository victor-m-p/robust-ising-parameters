import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt 
import seaborn as sns 

# setup 
n_timesteps = 100
n_visible, n_hidden = 16, 0
n_nodes = n_visible + n_hidden

# configurations 
questions = pd.read_csv(f'../data/preprocessing/questions_h{n_hidden}.csv')
probabilities = np.loadtxt(f'../data/preprocessing/prob_h{n_hidden}.txt')
configurations = np.loadtxt(f'../data/preprocessing/conf_h{n_hidden}.txt', dtype=int)

# get the files 
basepath = '../data/RCT/'
files = os.listdir(basepath)
files = [f for f in files if f.endswith('.txt')]
d = np.loadtxt(basepath + files[0])
n_timesteps = len(d)
d = pd.DataFrame({'fraction_outcome': d,
                  't': np.arange(n_timesteps)})

data_list = []
for f in files: 
    d = np.loadtxt(basepath + f)
    n_timesteps = len(d)
    d = pd.DataFrame({'fraction_outcome': d,
                      't': np.arange(n_timesteps)})
    d['condition'] = f.split('.')[0]
    d['n_hidden'] = f.split('.')[1]
    d['exp_type'] = f.split('.')[2]  
    d['enforce'] = f.split('.')[3]
    d['intervention_var'] = f.split('.')[4]
    d['outcome_var'] = f.split('.')[5]
    d['n_flips'] = f.split('.')[6]
    data_list.append(d)
d = pd.concat(data_list)   

# plot some stuff 
## (1) MDMA vs. placebo starting from NOT having major depression
d_prevention = d[d['exp_type'] == 'Prevention']
sns.lineplot(d_prevention['t'], 
             d_prevention['fraction_outcome'], 
             hue=d_prevention['condition'])

## (2) MDMA vs. placebo starting from HAVING major depression
d_treatment = d[d['exp_type'] == 'Treatment']
sns.lineplot(d_treatment['t'],
             d_treatment['fraction_outcome'],
             hue=d_treatment['condition'])