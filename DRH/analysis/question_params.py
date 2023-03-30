import pandas as pd 
import numpy as np 
import os 
import re
import itertools
import seaborn as sns 
import matplotlib.pyplot as plt 
import networkx as nx 

# files 
dir = '../data/sample_questions/mdl/'
param_files = [x for x in os.listdir(dir) if x.endswith('params.dat')]

# run loop 
hi_list = []
Jij_list = []
for file in param_files:
    n, i, j, samp = re.search(r"n(\d)_i(\d+)_j(\d+)_sample(\d+)", file).group(1, 2, 3, 4)
    n, i, j, samp = int(n), int(i), int(j), int(samp)
    nJ = int(n*(n-1)/2)
    #if i == focus_i and j == focus_j:
    params = np.loadtxt(dir + file)
    identifier = re.search(r"(.*?).dat_params", file).group(1)
    reference = pd.read_csv(f'../data/sample_questions/reference/{identifier}.csv')
    reference = reference.drop(columns = ['entry_id', 'weight']).columns.tolist()
    reference = [int(x) for x in reference]
    J = params[:nJ]
    h = params[nJ:]
    id = f"{i}_{j}"
    # hi 
    df_hi = pd.DataFrame({
        'id': [id for _, _ in enumerate(reference)],
        'n': [n for _, _ in enumerate(reference)],
        'sample': [samp for _, _ in enumerate(reference)],
        'q': reference,
        'h': h})
    # Jij
    df_Jij = pd.DataFrame([(i, j, J[num]) for num, (i, j) in enumerate(itertools.combinations(reference, 2))], columns=['i', 'j', 'coupling'])
    df_Jij['n'] = [n for _ in range(len(df_Jij))]
    df_Jij['sample'] = [samp for _ in range(len(df_Jij))]
    df_Jij['id'] = [id for _ in range(len(df_Jij))]
    # append 
    hi_list.append(df_hi)
    Jij_list.append(df_Jij)
df_hi = pd.concat(hi_list)
df_Jij = pd.concat(Jij_list)

# need to convert format from original question id to 1-20
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')
question_reference = question_reference.drop(columns = ['question_drh'])
recode_dict = question_reference[['question_id_drh', 'question_id']].set_index('question_id_drh')['question_id'].to_dict()

# read the 20-question reference system
hi_20 = pd.read_csv('../data/analysis/hi_20.csv')
Jij_20 = pd.read_csv('../data/analysis/Jij_20.csv')

# recode J and h
df_hi['q'] = df_hi['q'].replace(recode_dict)
df_hi['q'] = df_hi['q'].astype('category')

df_Jij['i'] = df_Jij['i'].replace(recode_dict)
df_Jij['j'] = df_Jij['j'].replace(recode_dict)

# plot overall hi
df_hi = df_hi.sort_values('q')
hi_20 = hi_20.sort_values('q')
fig, ax = plt.subplots(figsize = (3, 3), dpi = 300)
sns.boxplot(data=df_hi, x="h", y="q")
plt.plot(hi_20['h'], hi_20['q']-1, 'ro')
plt.savefig('../figures/hi_n5_test.pdf', bbox_inches = 'tight')

# work on overall Jij
# subset the Jij that we specifically ran samples for (n = 10)
df_Jij_sub = df_Jij.groupby(['i', 'j']).size().reset_index(name = 'count').sort_values('count', ascending = False).head(10)
df_Jij_sub = df_Jij.merge(df_Jij_sub, on = ['i', 'j'], how = 'inner')
Jij_20_sub = Jij_20.merge(df_Jij_sub[['i', 'j']].drop_duplicates(), on = ['i', 'j'], how = 'inner') 

df_Jij_sub['i, j'] = df_Jij_sub['i'].astype(str) + ', ' + df_Jij_sub['j'].astype(str)
Jij_20_sub['i, j'] = Jij_20_sub['i'].astype(str) + ', ' + Jij_20_sub['j'].astype(str)

df_Jij_sub = df_Jij_sub.sort_values('i, j')
Jij_20_sub = Jij_20_sub.sort_values('i, j')

fig, ax = plt.subplots(figsize = (3, 3), dpi = 300)
sns.boxplot(data = df_Jij_sub, x = 'coupling', y = 'i, j')
plt.plot(Jij_20_sub['coupling'], Jij_20_sub['i, j'], 'ro')
plt.savefig('../figures/Jij_n5_test.pdf', bbox_inches = 'tight')

# look at specific Jij (Monuments and Child Sacrifice)
df_Jij_sub_3_16 = df_Jij_sub[(df_Jij_sub['i'] == 3) & (df_Jij_sub['j'] == 16)]
Jij_20_sub_3_16 = Jij_20_sub[(Jij_20_sub['i'] == 3) & (Jij_20_sub['j'] == 16)]

# Jij plot 
fig, ax = plt.subplots(figsize = (5, 3), dpi = 300)
sns.histplot(df_Jij_sub_3_16['coupling'], ax = ax, label = 'n=5', bins = 100)
plt.suptitle('Monumental Child Sacrifice')
ax.vlines(Jij_20_sub_3_16['coupling'].values, 0, 1, color = 'red', label = 'n=20')
plt.show();

# find max and min 
max_systems = df_Jij_sub_3_16.sort_values('coupling', ascending = False)[['sample', 'id']].head(10) # 36, 15, 17, 20, 0
min_systems = df_Jij_sub_3_16.sort_values('coupling', ascending = True)[['sample', 'id']].head(10) # 35, 7, 79, 54, 81

# find these in the overall dataset 
hi_max = df_hi.merge(max_systems, on = ['sample', 'id'], how = 'inner')
hi_max.groupby('q').size().reset_index(name = 'count').sort_values('count', ascending = False)

hi_min = df_hi.merge(min_systems, on = ['sample', 'id'], how = 'inner')
hi_min.groupby('q').size().reset_index(name = 'count').sort_values('count', ascending = False)

# Jij plot with this observation
Jij_20[(Jij_20['i'] == 2) & (Jij_20['j'] == 3)] # 0.54
Jij_20[(Jij_20['i'] == 2) & (Jij_20['j'] == 16)] # -0.24

Jij_20[(Jij_20['i'] == 3) & (Jij_20['j'] == 20)] # 0.33 
Jij_20[(Jij_20['i'] == 16) & (Jij_20['j'] == 20)] # 0.34

# plot Jij for n=20 specifically focusing on neighbors of some 
def create_graph(edgelist, edgeattr, nodelist, nodeattr): 

    G = nx.from_pandas_edgelist(
        edgelist,
        'i',
        'j', 
        edge_attr=edgeattr)

    labeldict = {}
    for _, row in nodelist.iterrows():
        node_id = int(row['q'])
        labeldict[node_id] = node_id 
        for attr in nodeattr: 
            G.nodes[node_id][attr] = row[attr]

    return G, labeldict

focus_nodes = [3, 16]
Jij_20_focus = Jij_20[(Jij_20['i'].isin(focus_nodes)) | (Jij_20['j'].isin(focus_nodes))]
G, labeldict = create_graph(Jij_20_focus, ['coupling'], 
                            hi_20, ['h'])

seed = 1
cmap = plt.cm.coolwarm
pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")
cutoff_n = 20

# plot 
fig, ax = plt.subplots(figsize = (6, 6), facecolor = 'w', dpi = 500)
plt.axis('off')

size_lst = list(nx.get_node_attributes(G, 'h').values())
weight_lst = list(nx.get_edge_attributes(G, 'coupling').values())
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
    G, pos, 
    node_size = 600,
    node_color = size_lst, 
    edgecolors = 'black',
    linewidths = 0.5,
    cmap = cmap, vmin = vmin_n, vmax = vmax_n 
)
nx.draw_networkx_edges(
    G, pos,
    width = weight_abs, 
    edge_color = weight_lst, 
    alpha = 0.7, 
    edge_cmap = cmap, edge_vmin = vmin_e, edge_vmax = vmax_e)
nx.draw_networkx_labels(G, pos, font_size = 14, labels = labeldict)
plt.show();

# plot a positive and a negative system
focus_nodes = [3, 16]
max_system_Jij = df_Jij[(df_Jij['sample'] == 36) & (df_Jij['id'] == '4776_4827')]
max_system_hi = df_hi[(df_hi['sample'] == 36) &(df_hi['id'] == '4776_4827')]
max_system_Jij.sort_values(['i', 'j'], inplace = True)
max_system_hi.sort_values('q', inplace = True)
G, labeldict = create_graph(max_system_Jij, ['coupling'], 
                            max_system_hi, ['h'])

seed = 1
cmap = plt.cm.coolwarm
pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")

# plot 
fig, ax = plt.subplots(figsize = (3, 3), facecolor = 'w', dpi = 500)
plt.axis('off')

size_lst = list(nx.get_node_attributes(G, 'h').values())
weight_lst = list(nx.get_edge_attributes(G, 'coupling').values())

# vmin, vmax edges
vmax_e = np.max(list(np.abs(weight_lst)))
vmin_e = -vmax_e

# vmin, vmax nodes
vmax_n = np.max(list(np.abs(size_lst)))
vmin_n = -vmax_n

size_abs = [abs(x)*3000 for x in size_lst]
weight_abs = [abs(x)*15 for x in weight_lst]

nx.draw_networkx_nodes(
    G, pos, 
    node_size = 600,
    node_color = size_lst, 
    edgecolors = 'black',
    linewidths = 0.5,
    cmap = cmap, vmin = vmin_n, vmax = vmax_n 
)
nx.draw_networkx_edges(
    G, pos,
    width = weight_abs, 
    edge_color = weight_lst, 
    alpha = 0.7, 
    edge_cmap = cmap, edge_vmin = vmin_e, edge_vmax = vmax_e)
nx.draw_networkx_labels(G, pos, font_size = 14, labels = labeldict)
plt.show();

# min system
focus_nodes = [3, 16]
max_system_Jij = df_Jij[(df_Jij['sample'] == 35) & (df_Jij['id'] == '4745_5137')]
max_system_hi = df_hi[(df_hi['sample'] == 35) &(df_hi['id'] == '4745_5137')]
max_system_Jij.sort_values(['i', 'j'], inplace = True)
max_system_hi.sort_values('q', inplace = True)
G, labeldict = create_graph(max_system_Jij, ['coupling'], 
                            max_system_hi, ['h'])

seed = 1
cmap = plt.cm.coolwarm
pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")

# plot 
fig, ax = plt.subplots(figsize = (3, 3), facecolor = 'w', dpi = 500)
plt.axis('off')

size_lst = list(nx.get_node_attributes(G, 'h').values())
weight_lst = list(nx.get_edge_attributes(G, 'coupling').values())

# vmin, vmax edges
vmax_e = np.max(list(np.abs(weight_lst)))
vmin_e = -vmax_e

# vmin, vmax nodes
vmax_n = np.max(list(np.abs(size_lst)))
vmin_n = -vmax_n

size_abs = [abs(x)*3000 for x in size_lst]
weight_abs = [abs(x)*15 for x in weight_lst]

nx.draw_networkx_nodes(
    G, pos, 
    node_size = 600,
    node_color = size_lst, 
    edgecolors = 'black',
    linewidths = 0.5,
    cmap = cmap, vmin = vmin_n, vmax = vmax_n 
)
nx.draw_networkx_edges(
    G, pos,
    width = weight_abs, 
    edge_color = weight_lst, 
    alpha = 0.7, 
    edge_cmap = cmap, edge_vmin = vmin_e, edge_vmax = vmax_e)
nx.draw_networkx_labels(G, pos, font_size = 14, labels = labeldict)
plt.show();


'''

# work on hi
df_hi_focus = df_hi[(df_hi['q'] == focus_i) | (df_hi['q'] == focus_j)]
df_hi_focus = df_hi_focus.replace({'q': recode_dict}) 

spirit_body = df_hi_focus[df_hi_focus['q'] == 4]
fig, ax = plt.subplots(figsize = (3, 3), dpi = 300)
sns.kdeplot(spirit_body['h'], label = 'n=5')
plt.suptitle('hi for spirit body distinction')
ax.vlines(hi_20_focus[hi_20_focus['q'] == 4]['h'].values, 0, 1, color = 'red', label = 'n=20')
plt.show();

punish = df_hi_focus[df_hi_focus['q'] == 13]
fig, ax = plt.subplots(figsize = (3, 3), dpi = 300)
sns.kdeplot(punish['h'], label = 'n=5')
plt.suptitle('hi for supernatural beings punish')
ax.vlines(hi_20_focus[hi_20_focus['q'] == 13]['h'].values, 0, 1, color = 'red', label = 'n=20')
plt.show();

# work on Jij
df_Jij_focus = df_Jij[(df_Jij['i'] == focus_i) & (df_Jij['j'] == focus_j)]
df_Jij_focus = df_Jij_focus.replace({'i': recode_dict, 'j': recode_dict})

# Jij 
fig, ax = plt.subplots(figsize = (5, 3), dpi = 300)
sns.kdeplot(df_Jij_focus['coupling'], ax = ax, label = 'n=5')
plt.suptitle('Jij for spirit body distinction and supernatural beings punish')
ax.vlines(Jij_20_focus['coupling'].values, 0, 3, color = 'red', label = 'n=20')
plt.show();

# dive deeper into hi
df_hi.sort_values('h', ascending = False).head(5)
df_Jij_max = df_Jij[df_Jij['sample'] == 47]
df_Jij_max = df_Jij_max.replace({'i': recode_dict, 'j': recode_dict})

# does not "hit" any of the ones that you'd expect
# for instance, 5 should be included following my thesis. 
# this is not generally the case. There are some recurrent
# ones, but in general it must be driven by more complex
# things, such as the Jijs. 

'''