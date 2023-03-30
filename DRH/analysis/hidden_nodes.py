import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D

# load hidden nodes data 
groundtruth = np.loadtxt('../data/hidden_nodes_10/Jh_questions_10_samples_500_scale_0.5.dat')
hidden0 = np.loadtxt('../data/hidden_nodes_10/questions_10_samples_500_scale_0.5_hidden_0.dat_params.dat')
hidden1 = np.loadtxt('../data/hidden_nodes_10/questions_10_samples_500_scale_0.5_hidden_1.dat_params.dat')
hiddenNone = np.loadtxt('../data/hidden_nodes_10/questions_10_samples_500_scale_0.5_hidden_NONE.dat_params.dat')
# look at h params 
n_nodes = 10
n_J = int(n_nodes*(n_nodes-1)/2)

# Jij
Jij_groundtruth = groundtruth[:n_J]
Jij_hidden0 = hidden0[:n_J]
Jij_hidden1 = hidden1[:n_J]
Jij_hiddenNone = hiddenNone[:n_J]

# h
h_groundtruth = groundtruth[n_J:]
h_hidden0 = hidden0[n_J:]
h_hidden1 = hidden1[n_J:]
h_hiddenNone = hiddenNone[n_J:]

# h & J params 
# a hidden node provides really strong regularization
clrs_dict = {'hidden0': 'tab:orange', 
             'hidden1': 'tab:red', 
             'hiddenNone': 'tab:blue'}

custom_lines = [Line2D([0], [0], color=clrs_dict.get('hidden0'), lw=4),
                Line2D([0], [0], color=clrs_dict.get('hidden1'), lw=4),
                Line2D([0], [0], color=clrs_dict.get('hiddenNone'), lw=4)]

fig, ax = plt.subplots() 
plt.scatter(h_groundtruth, h_hiddenNone, label='hiddenNone', color = 'tab:blue')
plt.scatter(h_groundtruth, h_hidden0, label='hidden0', color = 'tab:orange')
plt.scatter(h_groundtruth, h_hidden1, label='hidden1', color = 'tab:red')
plt.legend(custom_lines, [x for x in clrs_dict.keys()])
plt.plot([-2, 2], [-2, 2], 'k-')
plt.xlabel('ground truth')
plt.ylabel('inferred')
plt.suptitle('$h_i$ params')

plt.scatter(Jij_groundtruth, Jij_hiddenNone, label='hiddenNone', color = 'tab:blue')
plt.scatter(Jij_groundtruth, Jij_hidden0, label='hidden0', color = 'tab:orange')
plt.scatter(Jij_groundtruth, Jij_hidden1, label='hidden1', color = 'tab:red')
plt.plot([-2, 2], [-2, 2], 'k-')
plt.legend(custom_lines, [x for x in clrs_dict.keys()])
plt.xlabel('ground truth')
plt.ylabel('inferred')
plt.suptitle('$J_{ij}$ params')

# check the actual differences (h)
# both of them set the hidden node to approximately 0 
focus_node = 0
plt.scatter(h_hidden0[focus_node], 1, color = 'tab:orange')
plt.scatter(h_hidden1[focus_node], 1, color = 'tab:red')
plt.scatter(h_hiddenNone[focus_node], 1, color = 'tab:blue')
plt.scatter(h_groundtruth[focus_node], 1, color = 'black')

# check the actual differences (Jij)
import itertools 
n = 10 
Jij_ind = list(itertools.combinations([i for i in range(10)], 2))
d_Jij = pd.DataFrame({
    'Jij_groundtruth': Jij_groundtruth,
    'Jij_hidden0': Jij_hidden0,
    'Jij_hidden1': Jij_hidden1,
    'Jij_hiddenNone': Jij_hiddenNone,
    'Jij': Jij_ind, 
    'Ji': [i[0] for i in Jij_ind],
    'Jj': [i[1] for i in Jij_ind]
})
