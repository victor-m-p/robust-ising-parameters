import pandas as pd 
import numpy as np 
from fun import p_dist, bin_states, par_from_file, probs_from_file, states_from_n 
import matplotlib.pyplot as plt 

# setup
n_nodes = 9

# couplings / h 
params_subset = par_from_file('/home/vmp/robust-ising-parameters/DRH/data/robustness/time_mpf/pre_reformation.txt_params.dat', n_nodes)
params_full = par_from_file('/home/vmp/robust-ising-parameters/DRH/data/robustness/time_mpf/full_data.txt_params.dat', n_nodes)

def plot_params(p_true, p_infer): 
    lim_min = np.min([np.min(p_true), np.min(p_infer)])
    lim_max = np.max([np.max(p_true), np.max(p_infer)])
    plt.plot(p_true, p_infer, 'o', alpha=1)
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k-')
    plt.xlabel('true param')
    plt.ylabel('infer param')
    plt.show();
    
plot_params(params_full, params_subset)

p_subset = probs_from_file('/home/vmp/robust-ising-parameters/DRH/data/robustness/time_mpf/pre_reformation.txt_params.dat', n_nodes)
p_full = probs_from_file('/home/vmp/robust-ising-parameters/DRH/data/robustness/time_mpf/full_data.txt_params.dat', n_nodes)

# states / configurations 
def plot_states(p_true, p_infer):
    lim = np.max([np.max(p_true), np.max(p_infer)])
    plt.plot(p_true, p_infer, 'o', alpha=1)
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.plot([0, lim], [0, lim], 'k-')
    plt.xlabel('true p config')
    plt.ylabel('infer p config')
    plt.show();

plot_states(p_full, p_subset) 

# find the states that it get most wrong?
# too many states with essential 0 probability 
configurations = bin_states(n_nodes)

#def pct_diff(n1, n2): 
#    return abs((n1 - n2) / ((n1 + n2) / 2)) * 100

#def abs_diff(n1, n2): 
#    return abs(n1 - n2)

def diff(n1, n2): 
    return n1 - n2

#vect_pct_diff = np.vectorize(pct_diff)
#vect_abs_diff = np.vectorize(abs_diff)
vect_diff = np.vectorize(diff)

#pct_diff_states = vect_pct_diff(p_full, p_subset)
#abs_diff_states = vect_abs_diff(p_full, p_subset)
diff_states = vect_diff(p_full, p_subset)

# this should be read from the bottom
# i.e., last row largest difference 
#abs_diff_idx = np.argsort(abs_diff_states)[-5:]
#abs_diff_config = configurations[abs_diff_idx]

#pct_diff_idx = np.argsort(pct_diff_states)[-5:] 
#pct_diff_config = configurations[pct_diff_idx] # basically all [1, -1]

diff_under_idx = np.argsort(diff_states)[-5:]
diff_under_config = configurations[diff_under_idx]
diff_over_idx = np.argsort(diff_states)[5:]
diff_over_config = configurations[diff_over_idx]

# what in particular is it getting wrong
features = [
    '4676', # official political support
    '4729', # scriptures
    '4745', # monuments
    '4776', # spirit-body distinction
    '4787', # reincarnation in this world
    '4814', # grave goods
    '4821', # formal burials
    '4954', # monitoring
    '5152', # small-scale rituals
]

d_over = pd.DataFrame(diff_over_config, columns = features)
d_over['type'] = 'over'
d_under = pd.DataFrame(diff_under_config, columns = features)
d_under['type'] = 'under'
d_diff = pd.concat([d_over, d_under])

# which religions have these properties? 
full_data = pd.read_csv('../data/robustness/time_reference/full_data.csv')
entry_reference = pd.read_csv('../data/analysis/entry_reference.csv')
full_reference = full_data.merge(entry_reference, on='entry_id', how = 'inner')
difference = full_reference.merge(d_diff, on = features, how = 'inner')

difference[difference['type'] == 'under'] # yes: we do underestimate protestants
difference[difference['type'] == 'over'] # we overestimate a lot of things ... 

