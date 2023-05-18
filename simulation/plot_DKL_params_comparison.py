import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
import json 

# setup
nn = 13
nsim = 500
n = 100
figpath = 'fig/fully_connected/'
parampath = f'data/fully_connected_nn{nn}_nsim{nsim}_params/'

# load DKL  
with open(f"{parampath}DKL_observed_visible_n{n}.json", 'r') as f:
    DKL_visible = json.load(f)
    
with open(f"{parampath}DKL_observed_hidden_n{n}.json", 'r') as f:
    DKL_hidden = json.load(f)

# load parameters
with open(f'{parampath}/MSE_observed_hidden.json', 'r') as f:
    MSE_hidden = json.load(f)

with open(f'{parampath}/MSE_observed_visible.json', 'r') as f: 
    MSE_visible = json.load(f) 

## todf
# expand the dictionary
def dct_to_df(dct, val):
    expanded = [(k, i+1, v_i) for k, v in dct.items() for i, v_i in enumerate(v)]
    df = pd.DataFrame(expanded, columns=['idx', 'num', f'{val}'])
    return df 

MSE_hidden_df = dct_to_df(MSE_hidden, 'MSE')
MSE_visible_df = dct_to_df(MSE_visible, 'MSE')
DKL_hidden_df = dct_to_df(DKL_hidden, 'DKL')
DKL_visible_df = dct_to_df(DKL_visible, 'DKL')

# merge properly
d_hidden = pd.merge(MSE_hidden_df, DKL_hidden_df, on=['idx', 'num'], how='inner')
d_visible = pd.merge(MSE_visible_df, DKL_visible_df, on=['idx', 'num'], how='inner')

d_hidden['idx'] = d_hidden['idx'].astype(float)
d_hidden['sparsity'] = pd.cut(d_hidden['idx'], bins=[-np.inf, -0.3, 0.3, np.inf], 
                         labels=["[-1;-0.3[", "[-0.3, 0.3]", "]0.3;1]"])

sns.scatterplot(d_hidden['MSE'], 
                d_hidden['DKL'], 
                hue=d_hidden['sparsity'],
                alpha=0.5)

d_visible['idx'] = d_visible['idx'].astype(float)
d_visible['sparsity'] = pd.cut(d_visible['idx'], bins=[-np.inf, -0.3, 0.3, np.inf],
                                 labels=["[-1;-0.3[", "[-0.3, 0.3]", "]0.3;1]"])

sns.scatterplot(d_visible['MSE'],
                d_visible['DKL'],
                hue=d_visible['sparsity'],
                alpha=0.5)    