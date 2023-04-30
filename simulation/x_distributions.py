'''
VMP 2023-04-30:
Testing the correlation we get with different types of distributions
over the parameters. 
'''

import numpy as np 
from sample_functions import sample_not_connected, sample_hidden_connected, sample_fully_connected
import matplotlib.pyplot as plt 
import seaborn as sns 

# overall params (maximum 11 nodes) 
n_hidden = [1, 2, 3]
n_visible = [2, 4, 8]
n_simulations = 500 # ballpark for DRH data 

# loop over different combinations 
for n_hidden, n_visible in zip(n_hidden, n_visible):
    print(n_hidden)
    print(n_visible)

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

# but still very few ACTUALLY zero; 
# this still bothers me a bit actually. 
# how do I get this thing to give me zeros? 
sns.kdeplot(np.random.laplace(0, 0.5, 1000))




# sample from model that is independent in both layers
sim_not_connected = sample_not_connected(
    n_simulations, 
    h_hidden, 
    h_visible, 
    J_interlayer
    )

# from model that is connected in hidden layer but independent in visible layer
sim_hidden_connected = sample_hidden_connected(
    n_simulations,
    h_hidden,
    J_hidden,
    h_visible,
    J_interlayer
)

# sample from the model that is connected in both layers
sim_fully_connected = sample_fully_connected(
    n_simulations,
    h,
    J
)

# test correspondence (they should not correspond in this case)
## mean: all different
np.mean(sim_not_connected, axis=0)
np.mean(sim_hidden_connected, axis=0) 
np.mean(sim_fully_connected, axis=0) 
## correlation: all different 
np.corrcoef(sim_not_connected, rowvar=False)
np.corrcoef(sim_hidden_connected, rowvar=False)
np.corrcoef(sim_fully_connected, rowvar=False) # we can get nan here if all a node is always on or off

# convert to MPF format 

