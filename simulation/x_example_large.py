'''
VMP 2023-04-30:
tests on larger N where the independent model should be a lot faster.
this is generally true but with some caveats: 
(1) for really low n the fully connected model is fastest (e.g. < 5 nodes).
(2) for really low n hidden nodes (e.g. < 5) basically no difference between
the unconnected and the hidden connected model. 
(3) for larger n the fully connected model does become much slower. 
'''

import numpy as np 
from sample_functions import sample_not_connected, sample_hidden_connected, sample_fully_connected
import timeit 

# overall params
n_hidden = 12
n_visible = 3
n_simulations = 1000

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

# test speed 
## not connected 
timeit.timeit('sample_not_connected(n_simulations, h_hidden, h_visible, J_interlayer)', 
              number=10,
              globals=globals()) 
timeit.timeit('sample_hidden_connected(n_simulations, h_hidden, J_hidden, h_visible, J_interlayer)',
              number=10,
              globals=globals()) 
timeit.timeit('sample_fully_connected(n_simulations, h, J)',
              number=10,
              globals=globals()) 

''' 
not connected should always be fast.
hidden connected can get slow if there are many nodes in hidden layer
fully connected scales with total number of nodes in system so gets much slower
'''

