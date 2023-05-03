'''
MPF_CMU/sim_fully_connected.sh
'''

import re
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from plot_functions import plot_params, plot_h_hidden

# setup 
figpath = 'fig/fully_connected/'
path_mpf = 'data/fully_connected_mpf/'
path_true = 'data/fully_connected_true/'

# load mpf 
param_files = [x for x in os.listdir(path_mpf) if x.endswith('.dat')]
h_dict_mpf = {}
J_dict_mpf = {}
reg = r'sim_(\w+)_mpf_nhid_(\d+)_nvis_(\d+)_th_(\w+)_(\w+.\w+)_(\w+.\w+)_tj_(\w+)_(\w+.\w+)_(\w+.\w+)'
for filename in param_files: 
    condition, n_hidden, n_visible, hdist, hmean, hstd, Jdist, Jmean, Jstd = re.findall(reg, filename)[0]
    identifier = f'{condition}{n_hidden}{n_visible}{hdist}{hmean}{hstd}{Jdist}{Jmean}{Jstd}'
    n_nodes = int(n_hidden) + int(n_visible)
    n_connections = int(n_nodes*(n_nodes-1)/2)
    Jh = np.loadtxt(f'{path_mpf}{filename}')
    J = Jh[:n_connections]
    h = Jh[n_connections:]
    h_dict_mpf[identifier] = h
    J_dict_mpf[identifier] = J

h_hidden_std1 = h_dict_mpf['hid24gaussian0.01.0gaussian0.01.0']
h_visible_std1 = h_dict_mpf['vis06gaussian0.01.0gaussian0.01.0']
h_hidden_std05 = h_dict_mpf['hid24gaussian0.00.5gaussian0.00.5']
h_visible_std05 = h_dict_mpf['vis06gaussian0.00.5gaussian0.00.5']

# load the ground truth files 
param_files = [x for x in os.listdir(path_true) if x.startswith('format')]
reg = r'format_Jh_nhid_(\d+)_nvis_(\d+)_th_(\w+)_(\w+.\w+)_(\w+.\w+)_tj_(\w+)_(\w+.\w+)_(\w+.\w+)'
h_dict_true = {}
J_dict_true = {}
for filename in param_files: 
    n_hidden, n_visible, hdist, hmean, hstd, Jdist, Jmean, Jstd = re.findall(reg, filename)[0]
    identifier = f'{n_hidden}{n_visible}{hdist}{hmean}{hstd}{Jdist}{Jmean}{Jstd}'
    n_nodes = int(n_hidden) + int(n_visible)
    n_connections = int(n_nodes*(n_nodes-1)/2)
    Jh = np.loadtxt(f'{path_true}{filename}')
    J = Jh[:n_connections]
    h = Jh[n_connections:]
    h_dict_true[identifier] = h
    J_dict_true[identifier] = J

h_true_std1 = h_dict_true['06gaussian0.01.0gaussian0.01.0']
h_true_std05 = h_dict_true['06gaussian0.00.5gaussian0.00.5']

## error = 0.5
### h
#### is this error reasonable for the visible nodes?   
plot_params(h_true_std05, h_visible_std05, 'h_true vs. h_visible (std = 0.5)', 0.1) 
plt.savefig(f"{figpath}h_true_vs_h_visible_std05.png")
plt.close()

#### hard to compare directly because hidden nodes could be opposite assignment 
plot_h_hidden(h_true_std05, h_hidden_std05, 2, 'h_true vs. h_hidden (std = 0.5)', 0.1)
plt.savefig(f"{figpath}h_true_vs_h_hidden_std05.png")
plt.close()

## error = 1.0
# a lot of error now for std = 1.0
plot_params(h_true_std1, h_visible_std1, 'h_true vs. h_visible (std = 1.0)', 0.1)
plt.savefig(f"{figpath}h_true_vs_h_visible_std1.png")
plt.close()

# not even that much worse than the visible case 
plot_h_hidden(h_true_std1, h_hidden_std1, 2, 'h_true vs. h_hidden (std = 1.0)', 0.1)
plt.savefig(f"{figpath}h_true_vs_h_hidden_std1.png")
plt.close()

# why is visible so bad with std = 1.0 (just too large?)
# does not look like we can recover hidden reasonably (even with low error)

# J 
J_true = J_dict_true['06gaussian0.00.5gaussian0.00.5']
J_vis = J_dict_mpf['vis06gaussian0.00.5gaussian0.00.5']
J_hid = J_dict_mpf['hid24gaussian0.00.5gaussian0.00.5']

plot_params(J_true, J_vis, 'J_true vs. J_visible (std = 0.5)', 0.1)
plt.savefig(f"{figpath}J_true_vs_J_visible_std05.png")
plt.close()

plot_params(J_true, J_hid, 'J_true vs. J_hidden (std = 0.5)', 0.1) # hmmm 
plt.savefig(f"{figpath}J_true_vs_J_hidden_std05.png")
plt.close()