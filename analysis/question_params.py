import pandas as pd 
import numpy as np 
import os 
import re
import itertools
import seaborn as sns 
import matplotlib.pyplot as plt 

# files 
dir = '../data/sample_questions/mdl_input/'
param_files = [x for x in os.listdir(dir) if x.endswith('params.dat')]
param_files[0]

# i and j that we test  
focus_i = 4776
focus_j = 4983

# run loop 
hi_list = []
Jij_list = []
for file in param_files:
    n, i, j, samp = re.search(r"n(\d)_i(\d+)_j(\d+)_sample(\d+)", file).group(1, 2, 3, 4)
    n, i, j, samp = int(n), int(i), int(j), int(samp)
    nJ = int(n*(n-1)/2)
    if i == focus_i and j == focus_j:
        params = np.loadtxt(dir + file)
        identifier = re.search(r"(.*?).dat", file).group(1)
        reference = pd.read_csv(f'../data/sample_questions/reference/{identifier}.csv')
        reference = reference.drop(columns = ['entry_id', 'weight']).columns.tolist()
        reference = [int(x) for x in reference]
        J = params[:nJ]
        h = params[nJ:]
        # hi 
        df_hi = pd.DataFrame({
            'n': [n for _, _ in enumerate(reference)],
            'sample': [samp for _, _ in enumerate(reference)],
            'q': reference,
            'h': h})
        # Jij
        df_Jij = pd.DataFrame([(i, j, J[num]) for num, (i, j) in enumerate(itertools.combinations(reference, 2))], columns=['i', 'j', 'coupling'])
        df_Jij['n'] = [n for x in range(len(df_Jij))]
        df_Jij['sample'] = [samp for x in range(len(df_Jij))]
        # append 
        hi_list.append(df_hi)
        Jij_list.append(df_Jij)
df_hi = pd.concat(hi_list)
df_Jij = pd.concat(Jij_list)

# get hi and Jij from n=20 data 
def get_hi_Jij(n, corr_J, means_h): 
    nodes = [node+1 for node in range(n)]
    Jij = list(itertools.combinations(nodes, 2))
    Jij = pd.DataFrame(Jij, columns = ['i', 'j'])
    Jij['coupling'] = corr_J
    hi = pd.DataFrame(nodes, columns = ['q'])
    hi['h'] = means_h
    return hi, Jij

n_nodes, n_nan, n_rows, n_entries = 20, 5, 455, 407
A = np.loadtxt(f'../data/mdl_experiments/matrix_questions_{n_nodes}_maxna_{n_nan}_nrows_{n_rows}_entries_{n_entries}.txt.mpf_params.dat')
n_J = int(n_nodes*(n_nodes-1)/2)
J = A[:n_J] 
h = A[n_J:]
hi_20, Jij_20 = get_hi_Jij(n_nodes, J, h)

# subset the ones we need right now 
hi_20_focus = hi_20[(hi_20['q'] == 4) | (hi_20['q'] == 13)]
Jij_20_focus = Jij_20[(Jij_20['i'] == 4) & (Jij_20['j'] == 13)]

# need to convert format from original question id to 1-20
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')
question_reference = question_reference.drop(columns = ['question_drh'])

recode_dict = question_reference[['question_id_drh', 'question_id']].set_index('question_id_drh')['question_id'].to_dict()

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