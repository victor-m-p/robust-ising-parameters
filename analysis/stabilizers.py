import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# load configurations
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype = int)
configuration_probabilities = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')

# find the questions 
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')

# try official political support 
idx_pol_sup = np.where(configurations[:, 0] == 1)[0]
idx_not_pol_sup = np.where(configurations[:, 0] == -1)[0]
p_pol_sup = configuration_probabilities[idx_pol_sup]
p_not_pol_sup = configuration_probabilities[idx_not_pol_sup]

# first look overall 
p_pol_sup_norm = np.sort(p_pol_sup)[::-1] / np.max(p_pol_sup)
p_not_pol_sup_norm = np.sort(p_not_pol_sup)[::-1] / np.max(p_not_pol_sup)

n_states = len(p_pol_sup_norm)
plt.plot([np.log(i) for i in range(n_states)], 
         [np.log(i) for i in p_pol_sup_norm], color = 'tab:blue')
plt.plot([np.log(i) for i in range(n_states)],
         [np.log(i) for i in p_not_pol_sup_norm], color = 'tab:orange')

# stability for the top 1K states of each?
