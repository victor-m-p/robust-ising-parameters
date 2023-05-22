'''
log likelihood has been checked against mpf 
so we could just use this instead of calculating
would be a lot faster 
'''

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
from functools import reduce 
import os 

# setup
nn = 13
nsim = 500
n = 100
norm = 'l1'
figpath = f'fig/fully_connected_{norm}/'
parampath = f'data/fully_connected_nn{nn}_nsim{nsim}_{norm}_params/'

# create directory if does not exist
if not os.path.exists(figpath): 
    os.makedirs(figpath)

# load data
DKL_visible = pd.read_csv(f"{parampath}DKL_visible_n{n}.csv")
DKL_hidden = pd.read_csv(f"{parampath}DKL_hidden_n{n}.csv")
logl_visibile = pd.read_csv(f"{parampath}logL_visible_mpf.csv")
logl_hidden = pd.read_csv(f"{parampath}logL_hidden_mpf.csv")
param_magnitude_visible = pd.read_csv(f"{parampath}param_magnitude_visible.csv")
param_magnitude_hidden = pd.read_csv(f"{parampath}param_magnitude_hidden.csv")
param_MSE_visible = pd.read_csv(f"{parampath}param_MSE_visible.csv")
param_MSE_hidden = pd.read_csv(f"{parampath}param_MSE_hidden.csv")
logl_true = np.loadtxt(f"{parampath}logL_true.txt").reshape(1)[0]
true_magnitude = np.loadtxt(f"{parampath}param_magnitude_true.txt").reshape(1)[0]

# create super dataframes 
visible_dfs = [DKL_visible, logl_visibile, param_magnitude_visible, param_MSE_visible]
hidden_dfs = [DKL_hidden, logl_hidden, param_magnitude_hidden, param_MSE_hidden]

d_visible = reduce(lambda left,right: pd.merge(left,right,on=['idx', 'num'], how='inner'), visible_dfs)
d_hidden = reduce(lambda left,right: pd.merge(left,right,on=['idx', 'num'], how='inner'), hidden_dfs)

# add sparsity grouping 
d_hidden['idx'] = d_hidden['idx'].astype(float)
d_hidden['sparsity'] = pd.cut(d_hidden['idx'], bins=[-np.inf, -0.3, 0.3, np.inf],
                              labels=["[-1;-0.3[", "[-0.3, 0.3]", "]0.3;1]"])

d_visible['idx'] = d_visible['idx'].astype(float)
d_visible['sparsity'] = pd.cut(d_visible['idx'], bins=[-np.inf, -0.3, 0.3, np.inf],
                               labels=["[-1;-0.3[", "[-0.3, 0.3]", "]0.3;1]"])

# plot comparisons 
def plot_compare(d_hidden, d_visible, x_var, y_var, 
                 hlines=False, sharex=False, sharey=False,
                 figpath=False):
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), 
                            sharex=sharex, sharey=sharey)  # Create a grid of 1x2 subplots

    xmax_hid = d_hidden[f'{x_var}'].max()
    xmax_vis = d_visible[f'{x_var}'].max()

    if sharex: 
        xmax_hid = max(xmax_hid, xmax_vis)
        xmax_vis = xmax_hid

    # Plot 1
    sns.scatterplot(x=d_hidden[f'{x_var}'],
                    y=d_hidden[f'{y_var}'],
                    hue=d_hidden['sparsity'],
                    alpha=0.5,
                    ax=axs[0]) # Plot on first subplot
    if hlines: 
        axs[0].hlines(y=hlines, xmin=0, 
                      xmax=xmax_hid, color='tab:red', 
                      label='true logl')
    axs[0].set_title('with hidden nodes')

    # Plot 2
    sns.scatterplot(x=d_visible[f'{x_var}'],
                    y=d_visible[f'{y_var}'],
                    hue=d_visible['sparsity'],
                    alpha=0.5,
                    ax=axs[1]) # Plot on second subplot
    if hlines: 
        axs[1].hlines(y=hlines, xmin=0, 
                      xmax=xmax_vis, color='tab:red', 
                      label='true logl')
    axs[1].set_title('without hidden nodes')

    plt.tight_layout()  # Adjust subplot params so subplots don't overlap

    if not figpath: 
        plt.show()
    else: 
        plt.savefig(f"{figpath}{x_var}_{y_var}.png")

plot_compare(d_hidden, d_visible, 'DKL', 'logL', 
             hlines=logl_true, sharex=True, sharey=True,
             figpath=figpath)

plot_compare(d_hidden, d_visible, 'MSE', 'DKL',
             figpath=figpath)

plot_compare(d_hidden, d_visible, 'MSE', 'logL',
             hlines=logl_true, sharey=True,
             figpath=figpath)

plot_compare(d_hidden, d_visible, 'MSE', 'squared_magnitude',
             hlines=true_magnitude, figpath=figpath)

# wait; are we actually matching the hidden as well?
# I am not sure how we would evaluate this with the current setup.
# we would have to write the crazy function ...
# also double check that there is no bug, looks almost too similar. 

''' 
def construct_par_df(dct_magnitude, sparsity_range): 
    ## calculate metrics for squared parameters 
    hdi_prob = .95
    mean_magnitude = [(i, np.mean(dct_magnitude[i])) for i in sparsity_range]
    min_magnitude = [(i, np.min(dct_magnitude[i])) for i in sparsity_range]
    HDI_magnitude = [(i, az.hdi(np.array(dct_magnitude[i]), hdi_prob=hdi_prob)) for i in sparsity_range]
    HDI_magnitude = [(t[0], t[1][0], t[1][1]) for t in HDI_magnitude]

    ## gather metrics
    df_mean = pd.DataFrame(mean_magnitude, columns=['sparsity', 'mean_param_magnitude'])
    df_min = pd.DataFrame(min_magnitude, columns=['sparsity', 'min_param_magnitude'])
    df_HDI = pd.DataFrame(HDI_magnitude, columns=['sparsity', 'HDI_lower', 'HDI_upper'])
    df_magnitude = pd.merge(df_mean, df_HDI, on='sparsity')
    df_magnitude = pd.merge(df_magnitude, df_min, on='sparsity')
    df_magnitude['sparsity'] = df_magnitude['sparsity'].astype(float)
    
    return df_magnitude 



# overview of the magnitude of parameters
def plot_param_overview(df_magnitude, true_magnitude, n_nodes, n_sim, condition): 

    # main plot 
    fig, ax = plt.subplots(figsize=(10, 6))

    ## Vertical line for each X from Y_lower to Y_upper
    for _, row in df_magnitude.iterrows():
        plt.plot([row['sparsity'], row['sparsity']], 
                [row['HDI_lower'], row['HDI_upper']], 
                color='tab:grey')

    ## Scatter plot for mean_Y
    plt.scatter(df_magnitude['sparsity'], 
                df_magnitude['mean_param_magnitude'], 
                color='tab:blue', 
                label='mean(params**2)',
                zorder=2)

    ## Scatter plot for max_Y
    plt.scatter(df_magnitude['sparsity'], 
                df_magnitude['min_param_magnitude'], 
                color='tab:orange', 
                label='min(params**2)',
                zorder=2)

    ## horizontal line for true 
    plt.plot([-1, 1], [true_magnitude, true_magnitude], 
            color='tab:red', label='true params**2')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('mean(params**2) and min(params**2) with 95% HDI')
    plt.tick_params(axis='x', rotation=45)
    plt.xlabel('Sparsity')
    plt.ylabel(r'$\sum \; \beta^2$')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{figpath}param_magnitude_nn{n_nodes}_nsim{n_sim}_{condition}.png")

plot_param_overview(df_magnitude_hidden, true_magnitude, n_nodes, n_sim, 'hidden')
plot_param_overview(df_magnitude_visible, true_magnitude, n_nodes, n_sim, 'visible')
'''
