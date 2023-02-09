'''
VMP 2022-02-08: This is outdated
'''

import itertools 
import pandas as pd 
import matplotlib.pyplot as plt 
import networkx as nx 
import numpy as np
from fun import *

def node_edge_lst(n, corr_J, means_h): 
    nodes = [node+1 for node in range(n)]
    comb = list(itertools.combinations(nodes, 2))
    d_edgelst = pd.DataFrame(comb, columns = ['n1', 'n2'])
    d_edgelst['weight'] = corr_J
    d_nodes = pd.DataFrame(nodes, columns = ['n'])
    d_nodes['size'] = means_h
    d_nodes = d_nodes.set_index('n')
    dct_nodes = d_nodes.to_dict('index')
    return d_edgelst, dct_nodes

def create_graph(d_edgelst, dct_nodes,): 

    G = nx.from_pandas_edgelist(
        d_edgelst,
        'n1',
        'n2', 
        edge_attr=['weight', 'weight_abs'])

    # assign size information
    for key, val in dct_nodes.items():
        G.nodes[key]['size'] = val['size']

    # label dict
    labeldict = {}
    for i in G.nodes(): 
        labeldict[i] = i
    
    return G, labeldict

##### PLOT PARAMETERS ######
n_nodes, n_nan, n_rows = 20, 5, 455
A = np.loadtxt('q_20_nan_5.dat_params.dat') # the original one 
n_J = int(n_nodes*(n_nodes-1)/2)
J = A[:n_J] 
h = A[n_J:]

## make it 1-20, and cross-reference that with the related question IDs. 
d_edgelst, dct_nodes = node_edge_lst(n_nodes, J, h)
d_edgelst = d_edgelst.assign(weight_abs = lambda x: np.abs(x['weight']))

## try with thresholding 
G, labeldict = create_graph(d_edgelst, dct_nodes)

## subset graph based on edgelist 
### NB: still needs to not split the graph (does not currently)
### but this behavior should be guaranteed. 
H = G.copy()
d_edgelst_sub = d_edgelst[d_edgelst['weight_abs'] <= 0.15]
sub_edges = list(zip(d_edgelst_sub['n1'], d_edgelst_sub['n2']))
H.remove_edges_from(sub_edges) 

# setup 
seed = 1
cmap = plt.cm.coolwarm
cutoff_n = 15

# position
pos = nx.nx_agraph.graphviz_layout(H, prog = "fdp")

'''
## a few manual tweaks 
x, y = pos[16]
pos[16] = (x-25, y+0)
x, y = pos[7]
pos[7] = (x-10, y+0)
x, y = pos[1]
pos[1] = (x+5, y+5)
x, y = pos[4]
pos[4] = (x+25, y+25)
'''

## plot 
fig, ax = plt.subplots(figsize = (6, 6), facecolor = 'w', dpi = 500)
plt.axis('off')

size_lst = list(nx.get_node_attributes(H, 'size').values())
weight_lst = list(nx.get_edge_attributes(H, 'weight').values())
threshold = sorted([np.abs(x) for x in weight_lst], reverse=True)[cutoff_n]
weight_lst_filtered = [x if np.abs(x)>threshold else 0 for x in weight_lst]

# vmin, vmax edges
vmax_e = np.max(list(np.abs(weight_lst)))
vmin_e = -vmax_e

# vmin, vmax nodes
vmax_n = np.max(list(np.abs(size_lst)))
vmin_n = -vmax_n

size_abs = [abs(x)*3000 for x in size_lst]
weight_abs = [abs(x)*15 for x in weight_lst_filtered]

nx.draw_networkx_nodes(
    H, pos, 
    node_size = 600,#size_abs, 
    node_color = size_lst, 
    edgecolors = 'black',
    linewidths = 0.5,
    cmap = cmap, vmin = vmin_n, vmax = vmax_n 
)
nx.draw_networkx_edges(
    H, pos,
    width = weight_abs, 
    edge_color = weight_lst, 
    alpha = 0.7, # hmmm
    edge_cmap = cmap, edge_vmin = vmin_e, edge_vmax = vmax_e)
nx.draw_networkx_labels(G, pos, font_size = 14, labels = labeldict)
# add to axis
sm_edge = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin_e, vmax=vmax_e))
sm_edge._A = []
sm_node = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin_n, vmax=vmax_n))
sm_node._A = []
axis = plt.gca()
# maybe smaller factors work as well, but 1.1 works fine for this minimal example
#axis.set_xlim([1.1*x for x in axis.get_xlim()])
#axis.set_ylim([1.1*y for y in axis.get_ylim()])
plt.subplots_adjust(bottom=0.1, right=1, left=0, top=1)
#ax_edge = plt.axes([0.95, 0.12, 0.04, 0.74])
ax_edge = plt.axes([0.05, 0, 0.90, 0.05])
ax_node = plt.axes([0.05, -0.2, 0.9, 0.05])
plt.colorbar(sm_edge, cax = ax_edge, orientation='horizontal')
plt.colorbar(sm_node, cax = ax_node, orientation='horizontal')

#cbar.ax.yaxis.set_ticks_position('left') #yaxis.tick_left()
ax.text(0.24, -0.03, r'Pairwise couplings (J$_{ij}$)', size=20, transform=ax.transAxes)
ax.text(0.3, -0.25, r'Local fields (h$_i$)', size = 20, transform = ax.transAxes)
plt.savefig('figs/parameters_20.pdf', bbox_inches='tight')

###### 10 ######
ten_lst = sorted([2, 6, 8, 9, 10, 12, 13, 15, 16, 18])
n_nodes = 10
cutoff_n = 10
A = np.loadtxt('q_10_nan_5.dat_params.dat') 
n_J = int(n_nodes*(n_nodes-1)/2)
J = A[:n_J] 
h = A[n_J:]

G10 = G.subgraph(ten_lst)

# replace size 
for node, size in zip(ten_lst, h):
    G10.nodes[node]['size'] = size
G10.nodes(data=True)

# replace weight
d_edgelst, dct_nodes = node_edge_lst(n_nodes, J, h)
translation_dct = {idx+1:val for idx, val in enumerate(ten_lst)}
d_edgelst = d_edgelst.replace({'n1': translation_dct,
                               'n2': translation_dct})

for idx, row in d_edgelst.iterrows(): 
    edge_x = row['n1']
    edge_y = row['n2']
    weight = row['weight']
    weight_abs = abs(weight)
    G10.edges[(edge_x, edge_y)]['weight'] =  weight # does it create if not present?
    G10.edges[(edge_x, edge_y)]['weight_abs'] = weight_abs

labeldict10 = {}
for key, ele in labeldict.items(): 
    if key in ten_lst: 
        labeldict10[key] = ele 

## plot 
fig, ax = plt.subplots(figsize = (6, 6), facecolor = 'w', dpi = 500)
plt.axis('off')

size_lst = list(nx.get_node_attributes(G10, 'size').values())
weight_lst = list(nx.get_edge_attributes(G10, 'weight').values())
threshold = sorted([np.abs(x) for x in weight_lst], reverse=True)[cutoff_n]
weight_lst_filtered = [x if np.abs(x)>threshold else 0 for x in weight_lst]

# vmin, vmax edges
vmax_e = np.max(list(np.abs(weight_lst)))
vmin_e = -vmax_e

# vmin, vmax nodes
vmax_n = np.max(list(np.abs(size_lst)))
vmin_n = -vmax_n

size_abs = [abs(x)*3000 for x in size_lst]
weight_abs = [abs(x)*15 for x in weight_lst_filtered]

nx.draw_networkx_nodes(
    G10, pos, 
    node_size = 600,#size_abs, 
    node_color = size_lst, 
    edgecolors = 'black',
    linewidths = 0.5,
    cmap = cmap, vmin = vmin_n, vmax = vmax_n 
)
nx.draw_networkx_edges(
    G10, pos,
    width = weight_abs, 
    edge_color = weight_lst, 
    alpha = 0.7, # hmmm
    edge_cmap = cmap, edge_vmin = vmin_e, edge_vmax = vmax_e)
nx.draw_networkx_labels(G10, pos, font_size = 14, labels = labeldict10)
# add to axis
sm_edge = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin_e, vmax=vmax_e))
sm_edge._A = []
sm_node = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin_n, vmax=vmax_n))
sm_node._A = []
axis = plt.gca()
# maybe smaller factors work as well, but 1.1 works fine for this minimal example
#axis.set_xlim([1.1*x for x in axis.get_xlim()])
#axis.set_ylim([1.1*y for y in axis.get_ylim()])
plt.subplots_adjust(bottom=0.1, right=1, left=0, top=1)
#ax_edge = plt.axes([0.95, 0.12, 0.04, 0.74])
ax_edge = plt.axes([0.05, 0, 0.90, 0.05])
ax_node = plt.axes([0.05, -0.2, 0.9, 0.05])
plt.colorbar(sm_edge, cax = ax_edge, orientation='horizontal')
plt.colorbar(sm_node, cax = ax_node, orientation='horizontal')

#cbar.ax.yaxis.set_ticks_position('left') #yaxis.tick_left()
ax.text(0.24, -0.03, r'Pairwise couplings (J$_{ij}$)', size=20, transform=ax.transAxes)
ax.text(0.3, -0.25, r'Local fields (h$_i$)', size = 20, transform = ax.transAxes)
plt.savefig('figs/parameters_10.pdf', bbox_inches='tight')


### only five ###
five_lst = sorted([2, 6, 8, 10, 15])
n_nodes = 5
cutoff_n = 5
A = np.loadtxt('q_5_nan_5.dat_params.dat') 
n_J = int(n_nodes*(n_nodes-1)/2)
J = A[:n_J] 
h = A[n_J:]

G5 = G.subgraph(five_lst)

# replace size 
for node, size in zip(five_lst, h):
    G5.nodes[node]['size'] = size
G5.nodes(data=True)

# replace weight
d_edgelst, dct_nodes = node_edge_lst(n_nodes, J, h)
translation_dct = {idx+1:val for idx, val in enumerate(five_lst)}
d_edgelst = d_edgelst.replace({'n1': translation_dct,
                               'n2': translation_dct})

for idx, row in d_edgelst.iterrows(): 
    edge_x = row['n1']
    edge_y = row['n2']
    weight = row['weight']
    weight_abs = abs(weight)
    G5.edges[(edge_x, edge_y)]['weight'] =  weight # does it create if not present?
    G5.edges[(edge_x, edge_y)]['weight_abs'] = weight_abs

labeldict5 = {}
for key, ele in labeldict.items(): 
    if key in five_lst: 
        labeldict5[key] = ele 

## plot 
fig, ax = plt.subplots(figsize = (6, 6), facecolor = 'w', dpi = 500)
plt.axis('off')

size_lst = list(nx.get_node_attributes(G5, 'size').values())
weight_lst = list(nx.get_edge_attributes(G5, 'weight').values())
threshold = sorted([np.abs(x) for x in weight_lst], reverse=True)[cutoff_n]
weight_lst_filtered = [x if np.abs(x)>threshold else 0 for x in weight_lst]

# vmin, vmax edges
vmax_e = np.max(list(np.abs(weight_lst)))
vmin_e = -vmax_e

# vmin, vmax nodes
vmax_n = np.max(list(np.abs(size_lst)))
vmin_n = -vmax_n

size_abs = [abs(x)*3000 for x in size_lst]
weight_abs = [abs(x)*15 for x in weight_lst_filtered]

nx.draw_networkx_nodes(
    G5, pos, 
    node_size = 600,#size_abs, 
    node_color = size_lst, 
    edgecolors = 'black',
    linewidths = 0.5,
    cmap = cmap, vmin = vmin_n, vmax = vmax_n 
)
nx.draw_networkx_edges(
    G5, pos,
    width = weight_abs, 
    edge_color = weight_lst, 
    alpha = 0.7, # hmmm
    edge_cmap = cmap, edge_vmin = vmin_e, edge_vmax = vmax_e)
nx.draw_networkx_labels(G5, pos, font_size = 14, labels = labeldict5)
# add to axis
sm_edge = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin_e, vmax=vmax_e))
sm_edge._A = []
sm_node = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin_n, vmax=vmax_n))
sm_node._A = []
axis = plt.gca()
# maybe smaller factors work as well, but 1.1 works fine for this minimal example
#axis.set_xlim([1.1*x for x in axis.get_xlim()])
#axis.set_ylim([1.1*y for y in axis.get_ylim()])
plt.subplots_adjust(bottom=0.1, right=1, left=0, top=1)
#ax_edge = plt.axes([0.95, 0.12, 0.04, 0.74])
ax_edge = plt.axes([0.05, 0, 0.90, 0.05])
ax_node = plt.axes([0.05, -0.2, 0.9, 0.05])
plt.colorbar(sm_edge, cax = ax_edge, orientation='horizontal')
plt.colorbar(sm_node, cax = ax_node, orientation='horizontal')

#cbar.ax.yaxis.set_ticks_position('left') #yaxis.tick_left()
ax.text(0.24, -0.03, r'Pairwise couplings (J$_{ij}$)', size=20, transform=ax.transAxes)
ax.text(0.3, -0.25, r'Local fields (h$_i$)', size = 20, transform = ax.transAxes)
plt.savefig('figs/parameters_5.pdf', bbox_inches='tight')

