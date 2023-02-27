import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import os 
import re 
import itertools 
import seaborn as sns 

def parse_file(file, regex): 
    params = np.loadtxt(file)
    beta, type, idx = re.search(regex, file).group(1, 2, 3)
    idx = int(idx)
    n_params = len(params)
    n_nodes = int(0.5 * (np.sqrt(8*n_params+1)-1))
    n_J = int(n_nodes*(n_nodes-1)/2)
    Jij = params[:n_J]
    hi = params[n_J:]
    if type == 'removed': 
        n_questions = len(hi)+1
        combination_list = [x for x in range(n_questions) if x != idx]
    elif type == 'tworemoved': 
        n_questions = len(hi)+2
        combination_list = [x for x in range(n_questions) if x not in [idx, idx+1]]
    else: 
        n_questions = len(hi)
        combination_list = [x for x in range(n_questions)]
    Jij_idx = list(itertools.combinations(combination_list, 2))

    d_Jij = pd.DataFrame({
        'type': [type for _ in range(len(Jij))], 
        'id': [idx for _ in range(len(Jij))],
        'beta': [beta for _ in range(len(Jij))],
        'Jij': Jij_idx, 
        'Ji': [i[0] for i in Jij_idx],
        'Jj': [i[1] for i in Jij_idx],
        'coupling': Jij
    })
    
    d_hi = pd.DataFrame({
        'type': [type for _ in range(len(hi))], 
        'id': [idx for _ in range(len(hi))],
        'beta': [beta for _ in range(len(hi))],
        'hi': combination_list,
        'field': hi 
    })

    return d_Jij, d_hi 

def parse_true_file(file, regex): 
    params = np.loadtxt(file)
    beta = re.search(regex, file).group(1)
    type, idx = 'true', 0
    n_params = len(params)
    n_nodes = int(0.5 * (np.sqrt(8*n_params+1)-1))
    n_J = int(n_nodes*(n_nodes-1)/2)
    Jij = params[:n_J]
    hi = params[n_J:]
    n_questions = len(hi)
    combination_list = [x for x in range(n_questions)]
    Jij_idx = list(itertools.combinations(combination_list, 2))

    d_Jij = pd.DataFrame({
        'type': [type for _ in range(len(Jij))], 
        'id': [idx for _ in range(len(Jij))],
        'beta': [beta for _ in range(len(Jij))],
        'Jij': Jij_idx, 
        'Ji': [i[0] for i in Jij_idx],
        'Jj': [i[1] for i in Jij_idx],
        'coupling': Jij
    })
    
    d_hi = pd.DataFrame({
        'type': [type for _ in range(len(hi))], 
        'id': [idx for _ in range(len(hi))],
        'beta': [beta for _ in range(len(hi))],
        'hi': combination_list,
        'field': hi 
    })

    return d_Jij, d_hi 

def sns_scatter(data, param, lims): 
    sns.scatterplot(x='groundtruth', y=param,
                    hue='type', data=data)
    plt.plot(lims, lims, 'k-')

# setup
base_path = '../data/hidden_nodes/'
regex = r'questions_10_samples_500_scale_(\w+.\w+)_(\w+)_(\w+)'

# list files 
files = os.listdir('../data/hidden_nodes/')
files_infer = [f for f in files if f.endswith('.dat_params.dat')]
list_Jij, list_hi = [], []
for file in files_infer:
    d_Jij, d_hi = parse_file(os.path.join(base_path, file), regex)
    list_Jij.append(d_Jij)
    list_hi.append(d_hi)
d_Jij = pd.concat(list_Jij)
d_hi = pd.concat(list_hi)

# wrangle groundtruth & merge with data 
# here we actually do need to do something 
n_nodes = 10
n_J = int(n_nodes*(n_nodes-1)/2)
files_true = [f for f in files if 'true' in f]
list_Jij_true, list_hi_true = [], []
for file in files_true:
    d_Jij_true, d_hi_true = parse_file(os.path.join(base_path, file), regex)
    list_Jij_true.append(d_Jij_true)
    list_hi_true.append(d_hi_true)
d_Jij_true = pd.concat(list_Jij_true)
d_hi_true = pd.concat(list_hi_true)

d_Jij_true = d_Jij_true[['beta', 'Jij', 'coupling']].rename(columns = {'coupling': 'groundtruth'})
d_hi_true = d_hi_true[['beta', 'hi', 'field']].rename(columns = {'field': 'groundtruth'})

d_Jij_comb = pd.merge(d_Jij, d_Jij_true, on=['Jij', 'beta'])
d_hi_comb = pd.merge(d_hi, d_hi_true, on=['hi', 'beta'])

# assign whether observed or not 
d_Jij_comb['unobserved_lvl1'] = d_Jij_comb.apply(lambda row: str(row['id']) in str(row['Jij']) if row['type'] == 'hidden' else False, axis=1)
d_Jij_comb['unobserved_lvl2'] = d_Jij_comb.apply(lambda row: str(row['id']) in str(row['Jij']) or str(row['id']+1) in str(row['Jij']) if row['type'] == 'twohidden' else False, axis=1)
d_Jij_comb['observed'] = d_Jij_comb.apply(lambda row: True if row['unobserved_lvl1'] == False and row['unobserved_lvl2'] == False else False, axis=1)
d_Jij_obs = d_Jij_comb[d_Jij_comb['observed'] == True]

# some weird stuff here, need to check up on this 
d_Jij_comb_beta_low = d_Jij_comb[d_Jij_comb['beta'] == "0.2"]
d_Jij_obs_beta_low = d_Jij_obs[d_Jij_obs['beta'] == "0.2"]
sns_scatter(d_Jij_comb_beta_low, 'coupling', [-2, 2])
sns_scatter(d_Jij_obs_beta_low, 'coupling', [-0.7, 0.7])

d_Jij_comb_beta_med = d_Jij_comb[d_Jij_comb['beta'] == "0.5"]
d_Jij_obs_beta_med = d_Jij_obs[d_Jij_obs['beta'] == "0.5"]
sns_scatter(d_Jij_comb_beta_med, 'coupling', [-2, 2])
sns_scatter(d_Jij_obs_beta_med, 'coupling', [-1, 1])

# where is true here??
d_Jij_comb_beta_high = d_Jij_comb[d_Jij_comb['beta'] == "1.0"]
d_Jij_obs_beta_high = d_Jij_obs[d_Jij_obs['beta'] == "1.0"]
sns_scatter(d_Jij_comb_beta_high, 'coupling', [-2, 2])
sns_scatter(d_Jij_obs_beta_high, 'coupling', [-2, 2])

