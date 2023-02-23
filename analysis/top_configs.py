'''
Visualize the paratmers (Jij, hi) versus surface-level correlations. 
Produces figure 3A, 3B. 
VMP 2022-02-05: save .svg and .pdf
'''

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from fun import *
import seaborn as sns 

# load data
configuration_probabilities = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype = int)

# find the top configurations 
top_configuration_idx = np.argsort(configuration_probabilities)[::-1][:5]
top_configuration_p = configuration_probabilities[top_configuration_idx]
top_configurations = configurations[top_configuration_idx]
top_configuration_odds = [round(x*(2**20)) for x in top_configuration_p]

# question labels 
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')
question_labels = question_reference['question'].tolist()

# entry labels 
entry_maxlikelihood = pd.read_csv('../data/preprocessing/entry_maxlikelihood.csv')
entry_maxlikelihood = entry_maxlikelihood[entry_maxlikelihood['config_id'].isin(top_configuration_idx)]
entry_maxlikelihood = entry_maxlikelihood.sort_values('config_prob', ascending = False)

# plot 
fig, ax = plt.subplots(1, 1, figsize = (4, 7))
sns.heatmap(top_configurations.T, linewidth=5, cbar = False) # white = Y, black = N
plt.yticks(ticks = [x+0.5 for x in range(20)], 
           labels = question_labels, rotation = 0)
plt.xticks(ticks = [x+0.5 for x in range(5)],
           labels = ['Cistercians', 'Ancient Egypt', 'Jesuits', "Jehovah's Witnesses", 'Islam Aceh'],
           rotation = 45)

# draw circle 
import matplotlib.patches as patches
fig, ax = plt.subplots(1, 1, figsize = (4, 7))
sns.heatmap(top_configurations.T, linewidth=5, cbar = False) # white = Y, black = N
plt.yticks(ticks = [x+0.5 for x in range(20)], 
           labels = question_labels, rotation = 0)
plt.xticks(ticks = [x+0.5 for x in range(5)],
           labels = ['Cistercians', 'Ancient Egypt', 'Jesuits', "Jehovah's Witnesses", 'Islam Aceh'],
           rotation = 45)

for x in [0, 2, 6, 8, 19]: 
    rect = patches.Rectangle((0.05, x+0.05), 4.9, 0.9, 
                             linewidth=1, edgecolor='yellow', 
                             facecolor='yellow', alpha = 0.2)
    ax.add_patch(rect)

for x in [1, 3, 4, 9, 10, 11, 12, 17, 18]: 
    rect = patches.Rectangle((0.05, x+0.05), 4.9, 0.9, 
                             linewidth=1, edgecolor='tab:green', 
                             facecolor='tab:green', alpha = 0.2)
    ax.add_patch(rect)

for x in [5, 7, 13, 14, 15, 16]:
    rect = patches.Rectangle((0.05, x+0.05), 4.9, 0.9,
                             linewidth=1, edgecolor='tab:red',
                             facecolor='tab:red', alpha = 0.2)
    ax.add_patch(rect)
    
plt.show()
