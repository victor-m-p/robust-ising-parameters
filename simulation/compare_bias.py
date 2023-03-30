import pandas as pd 
import numpy as np 
from fun import p_dist, bin_states, par_from_file, probs_from_file, states_from_n 
import matplotlib.pyplot as plt 

# setup
n_nodes = 10

# couplings / h 
params_filter0 = par_from_file('data/bias_mpf/questions_10_samples_306_scale_0.5_filtered_pjsxasouda.dat_params.dat', n_nodes)
params_filter01 = par_from_file('data/bias_mpf/questions_10_samples_325_scale_0.5_filtered_pjsxasouda.dat_params.dat', n_nodes)
params_filter05 = par_from_file('data/bias_mpf/questions_10_samples_403_scale_0.5_filtered_pjsxasouda.dat_params.dat', n_nodes)
params_full = par_from_file('data/bias_mpf/questions_10_samples_500_scale_0.5_complete_pjsxasouda.dat_params.dat', n_nodes)
params_true = par_from_file('data/bias_true/questions_10_samples_500_scale_0.5_true_pjsxasouda.dat', n_nodes)

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
    
plot_params(params_true, params_full)
plot_params(params_true, params_filter0)
plot_params(params_true, params_filter01)
plot_params(params_true, params_filter05) # this looks almost perfect 

p_filter0 = probs_from_file('data/bias_mpf/questions_10_samples_306_scale_0.5_filtered_pjsxasouda.dat_params.dat', n_nodes)
p_filter01 = probs_from_file('data/bias_mpf/questions_10_samples_325_scale_0.5_filtered_pjsxasouda.dat_params.dat', n_nodes)
p_filter05 = probs_from_file('data/bias_mpf/questions_10_samples_403_scale_0.5_filtered_pjsxasouda.dat_params.dat', n_nodes)
p_full = probs_from_file('data/bias_mpf/questions_10_samples_500_scale_0.5_complete_pjsxasouda.dat_params.dat', n_nodes)
p_true = probs_from_file('data/bias_true/questions_10_samples_500_scale_0.5_true_pjsxasouda.dat', n_nodes)

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

plot_states(p_true, p_full) 
plot_states(p_true, p_filter0)
plot_states(p_true, p_filter01)
plot_states(p_true, p_filter05) # still not quite though 

# find the states that it get most wrong?
# too many states with essential 0 probability 
configurations = bin_states(n_nodes)

def pct_diff(n1, n2): 
    return abs((n1 - n2) / ((n1 + n2) / 2)) * 100

def abs_diff(n1, n2): 
    return abs(n1 - n2)

vect_pct_diff = np.vectorize(pct_diff)
vect_abs_diff = np.vectorize(abs_diff)

pct_diff_states = vect_pct_diff(p_true, p_filter0)
abs_diff_states = vect_abs_diff(p_true, p_filter0)

# this should be read from the bottom
# i.e., last row largest difference 
abs_diff_idx = np.argsort(abs_diff_states)[-10:]
configurations[abs_diff_idx]

pct_diff_idx = np.argsort(pct_diff_states)[-10:] 
configurations[pct_diff_idx] # basically all [1, -1]

########## the other observations ############

# couplings / h 
params_filter0 = par_from_file('/home/vmp/robust-ising-parameters/data/sim/bias_mpf/questions_10_samples_486_scale_0.5_filtered_rxquceoojx.dat_params.dat', n_nodes)
params_filter01 = par_from_file('/home/vmp/robust-ising-parameters/data/sim/bias_mpf/questions_10_samples_487_scale_0.5_filtered_rxquceoojx.dat_params.dat', n_nodes)
params_filter05 = par_from_file('/home/vmp/robust-ising-parameters/data/sim/bias_mpf/questions_10_samples_493_scale_0.5_filtered_rxquceoojx.dat_params.dat', n_nodes)
params_full = par_from_file('/home/vmp/robust-ising-parameters/data/sim/bias_mpf/questions_10_samples_500_scale_0.5_complete_rxquceoojx.dat_params.dat', n_nodes)
params_true = par_from_file('/home/vmp/robust-ising-parameters/data/sim/bias_true/questions_10_samples_500_scale_0.5_true_rxquceoojx.dat', n_nodes)

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
    
plot_params(params_true, params_full)
plot_params(params_true, params_filter0)
plot_params(params_true, params_filter01)
plot_params(params_true, params_filter05) 

p_filter0 = probs_from_file('/home/vmp/robust-ising-parameters/data/sim/bias_mpf/questions_10_samples_486_scale_0.5_filtered_rxquceoojx.dat_params.dat', n_nodes)
p_filter01 = probs_from_file('/home/vmp/robust-ising-parameters/data/sim/bias_mpf/questions_10_samples_487_scale_0.5_filtered_rxquceoojx.dat_params.dat', n_nodes)
p_filter05 = probs_from_file('/home/vmp/robust-ising-parameters/data/sim/bias_mpf/questions_10_samples_493_scale_0.5_filtered_rxquceoojx.dat_params.dat', n_nodes)
p_full = probs_from_file('/home/vmp/robust-ising-parameters/data/sim/bias_mpf/questions_10_samples_500_scale_0.5_complete_rxquceoojx.dat_params.dat', n_nodes)
p_true = probs_from_file('/home/vmp/robust-ising-parameters/data/sim/bias_true/questions_10_samples_500_scale_0.5_true_rxquceoojx.dat', n_nodes)

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

plot_states(p_true, p_full) 
plot_states(p_true, p_filter0)
plot_states(p_true, p_filter01)
plot_states(p_true, p_filter05)

## quantify distance 
