'''
MPF_CMU/sim_not_connected.sh
'''

import re
import os 
import numpy as np 

# extraction regex 
path_mpf = 'data/not_connected_mpf/'
param_files = [x for x in os.listdir(path_mpf) if x.endswith('.dat')]

for filename in param_files: 
    condition, hidden, visible, hdist, Jdist = re.findall(r'sim_(\w+)_mpf_nhid_(\d+)_nvis_(\d+)_th_(\w+)_tj_(\w+)', filename)[0]
    n_nodes = int(hidden) + int(visible)

# do it for an example first 
## first get the simulated params 
n_nodes = 6
nJ = int(n_nodes*(n_nodes-1)/2)

#### both hidden
filename = param_files[0]
condition, hidden, visible, hdist, Jdist = re.findall(r'sim_(\w+)_mpf_nhid_(\d+)_nvis_(\d+)_th_(\w+)_tj_(\w+)', filename)[0]
Jh = np.loadtxt(f'{path_mpf}{filename}')
J_hidden = Jh[:nJ]
h_hidden = Jh[nJ:]

#### both visible 
filename = param_files[2]
condition, hidden, visible, hdist, Jdist = re.findall(r'sim_(\w+)_mpf_nhid_(\d+)_nvis_(\d+)_th_(\w+)_tj_(\w+)', filename)[0]
Jh = np.loadtxt(f'{path_mpf}{filename}')
J_visible = Jh[:nJ]
h_visible = Jh[nJ:]

#### true params 
path_true = 'data/not_connected_true/'
hJ_true = np.loadtxt(f'{path_true}format_hJ_nhid_{hidden}_nvis_{visible}_th_{hdist}_tj_{Jdist}.txt')
h_true = hJ_true[:n_nodes]
J_true = hJ_true[n_nodes:]

h_true
h_visible # this should match almost perfectly 
h_hidden

# why is the visible so bad?
# too much noise?
# some error that I made?
# why particularly bad on the hidden nodes?

# check the means of these columns (looks fine I think)
A = np.loadtxt(f'{path_true}sim_true_nhid{hidden}_nvis_{visible}_th_{hdist}_tj_{Jdist}.txt')
A.mean(axis=0) # clearly favored, disfavored 

J_true
J_visible # generally not bad; but with some error...
J_hidden # really far off from ground truth 