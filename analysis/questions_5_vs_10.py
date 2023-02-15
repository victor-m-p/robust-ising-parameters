import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import re 
import os 
import itertools 

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
df_hi_5 = pd.concat(hi_list)
df_Jij_5 = pd.concat(Jij_list)


## load 10 
dir = '../data/sample_questions/mdl_10/'
param_files = [x for x in os.listdir(dir) if x.endswith('params.dat')]

# run loop 
hi_list = []
Jij_list = []
for file in param_files:
    n, i, j, samp = re.search(r"n(\d+)_i(\d+)_j(\d+)_sample(\d+)", file).group(1, 2, 3, 4)
    n, i, j, samp = int(n), int(i), int(j), int(samp)
    nJ = int(n*(n-1)/2)
    #if i == focus_i and j == focus_j:
    params = np.loadtxt(dir + file)
    identifier = re.search(r"(.*?).dat_params", file).group(1)
    reference = pd.read_csv(f'../data/sample_questions/reference_10/{identifier}.csv')
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
df_hi_10 = pd.concat(hi_list)
df_Jij_10 = pd.concat(Jij_list)

# find all of the questions we have specifically run for
h_pairs = [x.split("_") for x in df_hi_10.id.unique()]
i_list, j_list = [], []
for i, j in h_pairs: 
    i_list.append(int(i))
    j_list.append(int(j))
h_list = list(set(i_list + j_list))  

# subset these (not going to be completely balanced)
df_hi_10_sub = df_hi_10[(df_hi_10.q.isin(h_list))]
df_hi_5_sub = df_hi_5[(df_hi_5.q.isin(h_list))]
df_Jij_5_sub = df_Jij_5[(df_Jij_5.i.isin(h_list)) & (df_Jij_5.j.isin(h_list))]
df_Jij_10_sub = df_Jij_10[(df_Jij_10.i.isin(h_list)) & (df_Jij_10.j.isin(h_list))]

# need to convert format from original question id to 1-20
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')
question_reference = question_reference.drop(columns = ['question_drh'])
recode_dict = question_reference[['question_id_drh', 'question_id']].set_index('question_id_drh')['question_id'].to_dict()

# read the 20-question reference system
hi_20 = pd.read_csv('../data/analysis/hi_20.csv')
Jij_20 = pd.read_csv('../data/analysis/Jij_20.csv')

# recode J and h
df_hi_5_sub['q'] = df_hi_5_sub['q'].replace(recode_dict)
df_hi_5_sub['q'] = df_hi_5_sub['q'].astype('category')
df_hi_10_sub['q'] = df_hi_10_sub['q'].replace(recode_dict)
df_hi_10_sub['q'] = df_hi_10_sub['q'].astype('category')

df_Jij_5['i'] = df_Jij_5['i'].replace(recode_dict)
df_Jij_5['j'] = df_Jij_5['j'].replace(recode_dict)
df_Jij_10['i'] = df_Jij_10['i'].replace(recode_dict)
df_Jij_10['j'] = df_Jij_10['j'].replace(recode_dict)

# only a portion of the hi, Jij 
hi_list = df_hi_5_sub['q'].unique()
hi_20_sub = hi_20[hi_20['q'].isin(hi_list)]

### overall hi ### 

# concatenate hi 
df_hi_concat = pd.concat([df_hi_5_sub, df_hi_10_sub])

# plot overall hi
df_hi_5_sub = df_hi_5_sub.sort_values('q', ascending=True)
hi_20 = hi_20.sort_values('q')
fig, ax = plt.subplots(figsize = (3, 3), dpi = 300)
sns.boxplot(data=df_hi_concat, x="h", y="q", hue='n')
sns.scatterplot(data=hi_20_sub, x="h", y=[i for i in range(9)], 
                color = 'red', s = 20)
plt.savefig('../figures/hi_comparison_5_10.pdf', bbox_inches = 'tight')

### overall Jij ###

# concatenate Jij 
Jij_concat = ['4745_5137', '4676_5152', '4776_4827', '4776_4808', '4780_5220']
df_Jij_concat = pd.concat([df_Jij_5_sub, df_Jij_10_sub])
df_Jij_concat = df_Jij_concat[df_Jij_concat['id'].isin(Jij_concat)]

fig, ax = plt.subplots(figsize = (5, 5), dpi = 300)
sns.boxplot(data = df_Jij_concat, x = 'coupling', y = 'id', hue = 'n')
plt.savefig('../figures/Jij_comparison_5_10.pdf', bbox_inches = 'tight')

### sampling under a null model ### 
full_data = pd.read_csv('../data/reference/direct_reference_questions_20_maxna_5_nrows_455_entries_407.csv')

## correlation
d_corr = full_data[full_data['weight'] > 0.9999] # this is controversial
d_corr = d_corr.drop(columns = ['weight', 'entry_id'])
question_ids = d_corr.columns.tolist()
param_corr = d_corr.corr(method='pearson')
param_corr['i'] = param_corr.index 
param_corr_melt = pd.melt(param_corr, id_vars = 'i',
                          value_vars = question_ids,
                          value_name = 'correlation',
                          var_name = 'j')
param_corr_melt = param_corr_melt[param_corr_melt['i'] < param_corr_melt['j']]

## means 
d_mean = d_corr.mean().reset_index(name = 'mean')
d_mean = d_mean.rename(columns = {'index': 'q'})
d_mean['q'] = d_mean['q'].astype(int)
d_mean['q'] = d_mean['q'].replace(recode_dict)
d_mean['q'] = d_mean['q'].astype('category')
d_mean_sub = d_mean.merge(df_hi_5_sub[['q']].drop_duplicates(), on = 'q', how = 'inner')

###### plot means with hi ######
df_hi_5_sub = df_hi_5_sub.sort_values('q', ascending=True)
hi_20_sub = hi_20_sub.sort_values('q')
fig, ax = plt.subplots(figsize = (3, 3), dpi = 300)
sns.boxplot(data=df_hi_concat, x="h", y="q", hue='n')
sns.scatterplot(data=hi_20_sub, x="h", y=[i for i in range(9)], 
                color = 'red', s = 20)
sns.scatterplot(data=d_mean_sub, x='mean', y=[i for i in range(9)],
                color = 'black', s = 20)
plt.savefig('../figures/hi_comparison_5_10.pdf', bbox_inches = 'tight')

## correlations 
param_corr_melt
param_corr_melt['i'].replace(recode_dict)

