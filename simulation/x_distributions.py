'''
VMP 2023-04-30:
Testing the correlation we get with different types of distributions
over the parameters. 
'''

import numpy as np 
from sample_functions import sample_not_connected, sample_hidden_connected, sample_fully_connected
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 

##### 1. ALL GAUSSIAN #####

# overall params (maximum 11 nodes) 
n_hidden = 5
n_visible = 10
n_simulations = 1000 # ballpark for DRH data 

# set up data for the independent model
h_hidden = np.random.normal(0, 1, n_hidden) 
h_visible = np.random.normal(0, 1, n_visible)  
J_interlayer = np.random.normal(0, 1, (n_visible, n_hidden))

# corresponding data for the connected hidden layer (just adding J_hidden)
J_hidden = np.random.normal(0, 1, int(n_hidden*(n_hidden-1)/2)) 

# corresponding data for the fully connected model 
n_nodes = n_hidden + n_visible
h = np.random.normal(0, 1, n_nodes)
J = np.random.normal(0, 1, int(n_nodes*(n_nodes-1)/2))

# sample from model that is independent in both layers
sim_not_connected = sample_not_connected(
    n_simulations, 
    h_hidden, 
    h_visible, 
    J_interlayer
    )
sim_not_connected_visible = sim_not_connected[:, n_hidden:]

# from model that is connected in hidden layer but independent in visible layer
sim_hidden_connected = sample_hidden_connected(
    n_simulations,
    h_hidden,
    J_hidden,
    h_visible,
    J_interlayer
)
sim_hidden_connected_visible = sim_hidden_connected[:, n_hidden:]

# sample from the model that is connected in both layers
sim_fully_connected = sample_fully_connected(
    n_simulations,
    h,
    J
)
sim_fully_connected_visible = sim_fully_connected[:, n_hidden:]

# check means for visible layer 
h_mean_not_connected = np.mean(sim_not_connected_visible, axis=0) 
h_mean_hidden_connected = np.mean(sim_hidden_connected_visible, axis=0)
h_mean_fully_connected = np.mean(sim_fully_connected_visible, axis=0)

model = ['independent']*n_visible + ['hidden']*n_visible + ['fully']*n_visible
means = np.concatenate((h_mean_not_connected, h_mean_hidden_connected, h_mean_fully_connected))

d = pd.DataFrame({
    'model': model,
    'mean': means
})

fig, ax = plt.subplots()
sns.boxplot(data=d, x = 'mean', y='model', hue = 'model')
ax.get_legend().remove()
plt.show()

# check correlations between visible nodes 
## how do we extract what we actually need here?
corr_indices = np.triu_indices(n_visible, k =1)

corr_not_connected = np.corrcoef(sim_not_connected_visible, rowvar=False)
corr_not_connected = corr_not_connected[corr_indices]

corr_hidden_connected = np.corrcoef(sim_hidden_connected_visible, rowvar=False)
corr_hidden_connected = corr_hidden_connected[corr_indices]

corr_fully_connected = np.corrcoef(sim_fully_connected_visible, rowvar=False)
corr_fully_connected = corr_fully_connected[corr_indices]

model = ['independent']*int(n_visible*(n_visible-1)/2) + ['hidden']*int(n_visible*(n_visible-1)/2) + ['fully']*int(n_visible*(n_visible-1)/2)
correlations = np.concatenate((corr_not_connected, corr_hidden_connected, corr_fully_connected))

d = pd.DataFrame({
    'model': model,
    'corr': correlations
})

fig, ax = plt.subplots()
sns.boxplot(data=d, x = 'corr', y='model', hue = 'model')
ax.get_legend().remove()
plt.show()

'''
basically in the cases where we only have 
hidden -> visible connections, the correlations
can vanish (I think). I am not sure why this 
does not happen in the fully connected case. 
you would think that they would still tend to 
cancel out, but I guess the dynamics just become
really interesting in that case. 
'''

##### 1. h: Gaussian, J: Laplace #####
# we can look at this later
# ... 