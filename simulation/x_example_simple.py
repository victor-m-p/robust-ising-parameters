'''
VMP 2023-04-30:
shows different sampling implementations give the same results on equivalent data (as they should). 
'''

import numpy as np 
from sample_functions import sample_not_connected, sample_hidden_connected, sample_fully_connected
import timeit 

# set up data for the independent model
n_hidden = 2
n_visible = 3 
h_hidden = np.array([-0.5, 0.5])
h_visible = np.array([0, 0, 0]) 
J_interlayer = np.array([[1, 0], [0, -1], [0.5, 0]])

# corresponding data for the connected hidden layer (just adding J_hidden)
J_hidden = np.array([0]) 

nJ_hidden = int(n_hidden*(n_hidden-1)/2)
nJ_visible = int(n_visible*(n_visible-1)/2)
nJ_interlayer = n_hidden*n_visible

J_hidden = np.zeros(nJ_hidden)
J_visible = np.zeros(nJ_visible)
J_inter_flat = J_interlayer.flatten(order='F') # flatten column-major style

# think this works 
# we need to validate though
def construct_J(J_hidden, J_inter, J_visible, n_hidden, n_visible):
    idx_hidden = 0
    idx_inter = 0
    J_list = []
    for i in range(n_hidden):
        J_list += list(J_hidden[idx_hidden:idx_hidden+n_hidden-1])
        J_list += list(J_inter[idx_inter:idx_inter+n_visible])
        idx_hidden+=n_hidden-1
        idx_inter+=n_visible 

    J_list += list(J_visible)
    return J_list


def deconstruct_J(J, n_hidden, n_visible): 
    pass 


# did the other thing work as well?


# corresponding data for the fully connected model 
h = np.array([-0.5, # first hidden node 
              0.5, # second hidden node
              0, # first visible node
              0, # second visible node
              0 # third visible node 
              ])

J = np.array([0, # between two hidden 
              1, # first hidden to first non-hidden
              0, # first hidden to second non-hidden
              0.5, # first hidden to third non-hidden  
              0, # second hidden to first non-hidden
              -1, # second hidden to second non-hidden
              0, # second hidden to third non-hidden
              0, # first non-hidden to second non-hidden
              0, # first non-hidden to third non-hidden
              0 # second non-hidden to third non-hidden
              ])

# super parameters 
n_simulations = 10000

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

# test correspondence for BOTH hidden and visible layers
## mean: all the same 
np.mean(sim_not_connected, axis=0)
np.mean(sim_hidden_connected, axis=0) 
np.mean(sim_fully_connected, axis=0) 
## correlation: all the same
np.corrcoef(sim_not_connected, rowvar=False)
np.corrcoef(sim_hidden_connected, rowvar=False)
np.corrcoef(sim_fully_connected, rowvar=False) 

# test speed 
timeit.timeit('sample_not_connected(n_simulations, h_hidden, h_visible, J_interlayer)', 
              number=10,
              globals=globals()) # 2.2 seconds / iteration
timeit.timeit('sample_hidden_connected(n_simulations, h_hidden, J_hidden, h_visible, J_interlayer)',
              number=10,
              globals=globals()) # 2.1 second / iteration
timeit.timeit('sample_fully_connected(n_simulations, h, J)',
              number=10,
              globals=globals()) # 0.76 seconds / iteration (so faster for small n models)

import numpy as np 
from sample_functions import sample_not_connected, sample_hidden_connected, sample_fully_connected
import timeit 

# set up data for the independent model
n_hidden = 2
n_visible = 3 
h_hidden = np.array([-0.5, 0.5])
J_visible = 0 # independent visible layer  
h_visible = np.array([0, 0, 0]) 
J_interlayer = np.array([[1, 0], [0, -1], [0.5, 0]])

# corresponding data for the connected hidden layer (just adding J_hidden)
J_hidden = np.array([0]) 

# corresponding data for the fully connected model 
h = np.array([-0.5, # first hidden node 
              0.5, # second hidden node
              0, # first visible node
              0, # second visible node
              0 # third visible node 
              ])

J = np.array([0, # between two hidden 
              1, # first hidden to first non-hidden
              0, # first hidden to second non-hidden
              0.5, # first hidden to third non-hidden  
              0, # second hidden to first non-hidden
              -1, # second hidden to second non-hidden
              0, # second hidden to third non-hidden
              0, # first non-hidden to second non-hidden
              0, # first non-hidden to third non-hidden
              0 # second non-hidden to third non-hidden
              ])

# super parameters 
n_simulations = 10000

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

# test correspondence
## mean: all the same 
np.mean(sim_not_connected, axis=0)
np.mean(sim_hidden_connected, axis=0) 
np.mean(sim_fully_connected, axis=0) 
## correlation: all the same  
np.corrcoef(sim_not_connected, rowvar=False)
np.corrcoef(sim_hidden_connected, rowvar=False)
np.corrcoef(sim_fully_connected, rowvar=False) 

# test speed 
timeit.timeit('sample_not_connected(n_simulations, h_hidden, h_visible, J_interlayer)', 
              number=10,
              globals=globals()) # 1.96 seconds / iteration
timeit.timeit('sample_hidden_connected(n_simulations, h_hidden, J_hidden, h_visible, J_interlayer)',
              number=10,
              globals=globals()) # 2.04 second / iteration
timeit.timeit('sample_fully_connected(n_simulations, h, J)',
              number=10,
              globals=globals()) # 0.86 seconds / iteration (so faster for small n models)