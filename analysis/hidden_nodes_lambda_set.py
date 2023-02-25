import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import os 
import re 
import itertools 
import seaborn as sns 

# just test a couple 
groundtruth = np.loadtxt('../data/hidden_nodes_logs0/Jh_questions_8_samples_500_scale_0.5.dat')

# load hidden nodes data dynamical lambda 
files = os.listdir('../data/hidden_nodes_logs0/')
files = [f for f in files if f.endswith('.dat_params.dat')]
files_hidden = [f for f in files if 'hidden' in f]
files_removed = [f for f in files if 'removed' in f]

# focus on Jij for now 
Jij_list = []
hi_list = []
for file in files_hidden:
    params = np.loadtxt('../data/hidden_nodes_logs0/' + file)
    idx = re.search(r'questions_8_samples_500_scale_0.5_hidden_(\w+)', file).group(1)
    n_params = len(params)
    n_nodes = int(0.5 * (np.sqrt(8*n_params+1)-1))
    n_J = int(n_nodes*(n_nodes-1)/2)
    Jij = params[:n_J]
    hi = params[n_J:]
    Jij_ind = list(itertools.combinations([i for i in range(n_nodes)], 2))
    
    d_Jij = pd.DataFrame({
        'type': ['hidden' for i in range(len(Jij))],
        'id': idx,
        'Jij': Jij_ind, 
        'Ji': [i[0] for i in Jij_ind],
        'Jj': [i[1] for i in Jij_ind],
        'coupling': Jij
    })
    
    d_hi = pd.DataFrame({
        'type': ['hidden' for i in range(len(hi))],
        'id': idx,
        'hi': range(len(hi)),
        'field': hi 
    })

    Jij_list.append(d_Jij)
    hi_list.append(d_hi)

d_Jij_hidden = pd.concat(Jij_list)
d_hi_hidden = pd.concat(hi_list)

d_Jij_hidden

# check extra hi 
#hidden_hi = d_hi[d_hi['hi'] == 8] # very small
#d_hi_subset = d_hi[d_hi['hi'] != 8] # subset

# check extra Jij
#hidden_Jij = d_Jij[(d_Jij['Ji'] == 8) | (d_Jij['Jj'] == 8)]
#d_Jij_subset = d_Jij[(d_Jij['Ji'] != 8) & (d_Jij['Jj'] != 8)] # subset

## actually around the same size for the extra node system 
## so that one does not get shrunk out 
#np.mean([abs(x) for x in hidden_Jij['coupling']])
#np.mean([abs(x) for x in d_Jij_subset['coupling']])

n_nodes = 8
n_J = int(n_nodes*(n_nodes-1)/2)
groundtruthJ = groundtruth[:n_J]
Jij_ind = list(itertools.combinations([i for i in range(n_nodes)], 2))
d_groundtruth_Jij = pd.DataFrame({
    'Jij': Jij_ind,
    'groundtruth': groundtruthJ
})
#d_Jij_gt = d_groundtruth_Jij.merge(d_Jij_subset, on='Jij', how='inner')
#sns.scatterplot(x='groundtruth', y='coupling', hue='id', data=d_Jij_gt)
#plt.plot([-1.1, 1.1], [-1.1, 1.1], 'k-')

## try the other way around ...
Jij_list = []
hi_list = []
for file in files_removed:
    params = np.loadtxt('../data/hidden_nodes/' + file)
    idx = re.search(r'questions_8_samples_500_scale_0.5_removed_(\w+)', file).group(1)
    n_params = len(params)
    n_nodes = int(0.5 * (np.sqrt(8*n_params+1)-1))
    n_J = int(n_nodes*(n_nodes-1)/2)
    Jij = params[:n_J]
    hi = params[n_J:]
    Jij_ind = list(itertools.combinations([i for i in range(n_nodes)], 2))
    
    d_Jij = pd.DataFrame({
        'type': ['removed' for i in range(len(Jij))],
        'id': idx,
        'Jij': Jij_ind, 
        'Ji': [i[0] for i in Jij_ind],
        'Jj': [i[1] for i in Jij_ind],
        'coupling': Jij
    })
    
    d_hi = pd.DataFrame({
        'type': ['removed' for i in range(len(hi))],
        'id': idx,
        'hi': range(len(hi)),
        'field': hi 
    })

    Jij_list.append(d_Jij)
    hi_list.append(d_hi)

d_Jij_removed = pd.concat(Jij_list)
d_hi_removed = pd.concat(hi_list)


# hidden has these weird cases 
d_Jij_hidden['id'].unique()
d_Jij_hidden['type'] = [row['id'] if row['id'] in (['NONE', 'extra']) else row['type'] for i, row in d_Jij_hidden.iterrows()]
d_Jij_overall = pd.concat([d_Jij_removed, d_Jij_hidden])
d_Jij_gt = d_groundtruth_Jij.merge(d_Jij_overall, on='Jij', how='inner')

sns.scatterplot(x='groundtruth', y='coupling', hue='type', data=d_Jij_gt)
plt.plot([-1.5, 1.5], [-1.5, 1.5], 'k-')

# biggest deviations
d_Jij_gt['deviation'] = [abs(row['groundtruth'] - row['coupling']) for i, row in d_Jij_gt.iterrows()]
d_Jij_hid = d_Jij_gt[d_Jij_gt['type'] == 'hidden']
d_Jij_hid.sort_values('deviation', ascending=False).head(10)

# new crazy idea: 
# is the order any good? 
for i in range(8): 
    dns = d_Jij_gt[(d_Jij_gt['Ji'] != i) & (d_Jij_gt['Jj'] != i)]
    dns = dns[dns['id'] == f'{i}']
    sns.scatterplot(x='groundtruth', y='coupling', hue='type', data=dns)
    plt.plot([-1.5, 1.5], [-1.5, 1.5], 'k-')
    
# new crazy idea: 
clrs_dict = {'ground truth': 'black', 
             'inferred': 'tab:blue',
             'removed': 'tab:red', 
             'hidden': 'tab:orange'}

custom_lines = [Line2D([0], [0], color=clrs_dict.get('ground truth'), lw=4),
                Line2D([0], [0], color=clrs_dict.get('inferred'), lw=4),
                Line2D([0], [0], color=clrs_dict.get('removed'), lw=4),
                Line2D([0], [0], color=clrs_dict.get('hidden'), lw=4)]

for i in range(8):
    dns = d_Jij_gt[(d_Jij_gt['Ji'] != i) & (d_Jij_gt['Jj'] != i)]
    dns_i = dns[dns['id'] == f'{i}']
    dns_removed = dns_i[dns_i['type'] == 'removed']
    dns_removed_Jij = dns_removed['Jij']
    
    dns_hidden = dns_i[dns_i['type'] == 'hidden']
    dns_inferred = dns[dns['type'] == 'NONE']
    
    dns_hidden = dns_hidden.merge(dns_removed_Jij, on = 'Jij', how = 'inner')
    dns_inferred = dns_inferred.merge(dns_removed_Jij, on = 'Jij', how = 'inner')
    
    dns_hidden = dns_hidden.sort_values('groundtruth', ascending=False)
    dns_removed = dns_removed.sort_values('groundtruth', ascending=False)
    dns_inferred = dns_inferred.sort_values('groundtruth', ascending=False)
    
    plt.plot(dns_hidden['groundtruth'], range(len(dns_hidden)), clrs_dict.get('ground truth'))
    plt.plot(dns_inferred['coupling'], range(len(dns_inferred)), clrs_dict.get('inferred'))
    plt.plot(dns_removed['coupling'], range(len(dns_removed)), clrs_dict.get('removed'))
    plt.plot(dns_hidden['coupling'], range(len(dns_hidden)), clrs_dict.get('hidden'))
    plt.legend(custom_lines, [x for x in clrs_dict.keys()])
    plt.xlabel('coupling')
    plt.ylabel('parameter rank')
    plt.suptitle(f'hidden node {i+1}')
    plt.savefig(f'../figures/hidden_nodes/comparison_{i+1}.png')
    plt.gca().clear()

