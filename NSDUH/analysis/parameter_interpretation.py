'''
VMP 2023-03-24. 
Visualize parameters
'''

import pandas as pd 
import matplotlib.pyplot as plt 
import networkx as nx 
import numpy as np
import itertools 
from fun import *
import matplotlib.lines as mlines

# setup
n_visible, n_hidden = 16, 0
n_nodes = n_visible + n_hidden

# params 
A = np.loadtxt(f'../data/mdl/hidden_{n_hidden}.txt', dtype=float, delimiter=',')
n_J = int(n_nodes*(n_nodes-1)/2)
J = A[:n_J] 
h = A[n_J:]

# questions
questions = pd.read_csv(f'../data/preprocessing/questions_h{n_hidden}.csv')
questions = questions[['question', 'question_id']]

# gather this into dataframes 
def get_hi_Jij(n, corr_J, means_h):
    nodes = range(1, n+1)
    Jij = pd.DataFrame(list(itertools.combinations(nodes, 2)), columns=['i', 'j'])
    Jij['coupling'] = corr_J
    hi = pd.DataFrame({'question_id': nodes, 'h': means_h})
    return hi, Jij
hi, Jij = get_hi_Jij(n_nodes, J, h)

## hi
hi = hi.merge(questions, on='question_id', how='inner')

## Jij 
questions_i = questions.rename(columns={'question_id': 'i',
                                        'question': 'question_i'})
questions_j = questions.rename(columns={'question_id': 'j',
                                        'question': 'question_j'})
Jij = Jij.merge(questions_i, on='i', how='inner')
Jij = Jij.merge(questions_j, on='j', how='inner')

# list of all predictors 
predictors = ['Cigarette', 'Alcohol', 'Marijuana', 'Cocaine',
              'Heroin', 'LSD', 'PCP', 'Ecstasy', 'Inhalants',
              'Pain relievers', 'Tranquilizers', 'Stimulants',
              'Sedatives']
outcome = ['Psychological distress', 'Major depression', 'Suicide thoughts']

Jij_distress = Jij[Jij['question_i'].isin(predictors) & Jij['question_j'].isin(outcome)]
Jij_index = Jij_distress[Jij_distress['question_j'] == 'Psychological distress'].sort_values('coupling', ascending=False)
Jij_index = Jij_index[['question_i']].reset_index(drop=True)
Jij_index['index'] = Jij_index.index
Jij_distress = Jij_distress.merge(Jij_index, on='question_i', how='inner')
Jij_distress = Jij_distress.sort_values('index')

# Create scatterplot
fig, ax = plt.subplots()

# Customize y-axis tick labels
ax.set_yticks(Jij_distress['index'].unique())
ax.set_yticklabels(Jij_distress['question_i'].unique())

# Set axis labels
ax.set_xlabel('Coupling')
ax.set_ylabel('Predictor')

# Define color mapping for groups
group_colors = {'Suicide thoughts': 'tab:red', 
                'Major depression': 'tab:orange', 
                'Psychological distress': 'tab:green'}

# Define y offsets for each group
group_offsets = {'Suicide thoughts': -0.2, 
                 'Major depression': 0, 
                 'Psychological distress': 0.2}

# Iterate through the DataFrame rows and plot points
for index, row in Jij_distress.iterrows():
    color = group_colors[row['question_j']]
    y_offset = group_offsets[row['question_j']]
    ax.scatter(row['coupling'], row['index'] + y_offset, color=color)
    ax.plot([0, row['coupling']], [row['index'] + y_offset, row['index'] + y_offset], color=color)
# Show plot
legend_elements = [mlines.Line2D([], [], color=group_colors[group], marker='o', linestyle='', markersize=8, label=group)
                   for group in group_colors.keys()]

# Add legend to the plot
ax.legend(handles=legend_elements, loc='best', title='Groups')

plt.suptitle('Mental Health <-- Drug Usage')
plt.show()