import numpy as np 
import pandas as pd 
import re 
import os 
from sample_functions import read_text_file 

# match the files
figpath = 'fig/fully_connected/'
path_mpf = 'data/fully_connected_mpf/'
path_true = 'data/fully_connected_true/'

# load mpf 
param_files = [x for x in os.listdir(path_mpf) if x.endswith('.dat')]

# try to read something
filename = '/home/vmp/robust-ising-parameters/simulation/data/fully_connected_mpf/sim_vis_mpf_nhid_0_nvis_5_th_gaussian_0.0_0.1_tj_gaussian_0.0_0.1_nsim_500.txt_output_xIl0pwVD'

params, logl = read_text_file(filename)
params
logl



# 
J_true = np.array([]) 
h_true = np.array([])

def param_possibilities(params): 
    