import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 

# setup
nn = 13
nsim = 500
n = 100
figpath = 'fig/fully_connected/'
parampath = f'data/fully_connected_nn{nn}_nsim{nsim}_params/'

# load files
df_DKL_visible = pd.read_csv(f"{parampath}DKL_visible_n{n}.csv")
df_DKL_hidden = pd.read_csv(f"{parampath}DKL_hidden_n{n}.csv")

# plot mean(HDI) and min(HDI)
def plot_DKL_overview(df_DKL, n_nodes, n_sim, condition): 

    fig, ax = plt.subplots(figsize=(10, 6))

    # Vertical line for each X from Y_lower to Y_upper
    for _, row in df_DKL.iterrows():
        plt.plot([row['sparsity'], row['sparsity']], 
                [row['HDI_lower'], row['HDI_upper']], 
                color='tab:grey')

    # Scatter plot for mean_Y
    plt.scatter(df_DKL['sparsity'], 
                df_DKL['mean_DKL'], 
                color='tab:blue', 
                label='mean(DKL)',
                zorder=2)

    # Scatter plot for max_Y
    plt.scatter(df_DKL['sparsity'], 
                df_DKL['min_DKL'], 
                color='tab:orange', 
                label='min(DKL)',
                zorder=2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('mean(HDI) and min(HDI) with 95% HDI')
    plt.tick_params(axis='x', rotation=45)
    plt.xlabel('Sparsity')
    plt.ylabel(r'$D_{KL}(P_{true}||P_{model})$')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{figpath}DKL_L2_nn{n_nodes}_nsim{n_sim}_{condition}_overview.png")
    plt.close()

## does not appear better 
plot_DKL_overview(df_DKL_hidden, nn, nsim, 'hidden')
plot_DKL_overview(df_DKL_visible, nn, nsim, 'visible')

# plot them against each other: 
def plot_DKL_comparison(df_DKL_hidden, df_DKL_visible, param,
                        n_nodes, n_sim):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot for mean_Y
    plt.scatter(df_DKL_visible['sparsity'], 
                df_DKL_visible[f'{param}'], 
                color='tab:blue', 
                label=f'{param} visible',
                zorder=2)

    # Scatter plot for max_Y
    plt.scatter(df_DKL_hidden['sparsity'], 
                df_DKL_hidden[f'{param}'], 
                color='tab:orange', 
                label=f'{param} hidden',
                zorder=2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Comparison hidden vs. visible')
    plt.tick_params(axis='x', rotation=45)
    plt.xlabel('Sparsity')
    plt.ylabel(r'$D_{KL}(P_{true}||P_{model})$')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{figpath}DKL_L2_nn{n_nodes}_nsim{n_sim}_{param}_comparison.png")
    plt.close()

# can actually find better solutions (sometimes)
# could we have discovered these without DKL (i.e. LogL). 
# this also assumes that we know the correct number of hidden nodes.
# we have not shown this for a "wrong" number of hidden nodes. 
plot_DKL_comparison(df_DKL_hidden, df_DKL_visible, 'mean_DKL', nn, nsim)
plot_DKL_comparison(df_DKL_hidden, df_DKL_visible, 'min_DKL', nn, nsim)

# plot distributions 
## need to save distributions 
def plot_DKL_distributions(dct_DKL, n_nodes, n_sim, condition): 

    fig, ax = plt.subplots()
    valrange=["-1.00", "-0.50", "00.00", "00.50"]
    col=['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i in valrange:
        sns.histplot(dct_DKL[i], label=i, bins=10, color=col[valrange.index(i)])
    plt.xlabel(r'$D_{KL}(P_{true}||P_{model})$')
    plt.title('Distribution of DKL')
    plt.legend()
    plt.savefig(f"{figpath}DKL_L2_nn{n_nodes}_nsim{n_sim}_{condition}_distributions.png")
    plt.close()

## basically no variation in DKL. 
## is this just because there is a better way to fit the observed data
## than the ACTUAL parameters?
plot_DKL_distributions(dct_hidden_DKL)
plot_DKL_distributions(dct_visible_DKL)