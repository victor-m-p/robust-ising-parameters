import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import re
from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], marker='o', color='tab:blue', markersize=5),
                Line2D([0], [0], marker='o', color='tab:orange', markersize=5)]

# across all of them 
filenames = os.listdir('../data/logistic/')
filenames = [x for x in filenames if x.endswith('restrict.csv')]
filelist = []
for file in filenames: 
    n_questions, n_samples, beta = re.search(r'(\d+)_(\d+)_(\d+.\d+)', file).groups()
    d = pd.read_csv(f'../data/logistic/{file}')
    d['n_questions'] = n_questions 
    d['n_samples'] = n_samples 
    d['true_beta'] = beta
    filelist.append(d)
d = pd.concat(filelist)

# plot overall true params versus inferred params 
plt.scatter(d['Actual'], d['MPF'], color = 'tab:blue', s = 5)
plt.scatter(d['Actual'], d['transform'], color = 'tab:orange', s = 5)
plt.xlabel('True Parameters')
plt.ylabel('Inferred Parameters (MPF vs. Logistic)')
plt.legend(custom_lines, ['MPF', 'Logistic'])
plt.plot([-3, 3], [-3, 3], 'k-')

# error patterns? 
## mean error 
mean_error_logistic = np.abs(d['transform']-d['Actual']).mean()
mean_error_mpf = np.abs(d['MPF']-d['Actual']).mean()
mean_error_logistic # 0.1611
mean_error_mpf # 0.1524 

## median error 
median_error_logistic = np.abs(d['transform']-d['Actual']).median()
median_error_mpf = np.abs(d['MPF']-d['Actual']).median()
median_error_logistic # 0.075
median_error_mpf # 0.08

## error for beta (Jij), error for alpha (hi)
d_alpha = d[d['param'].str.contains('alpha')] # more problematic
d_beta = d[d['param'].str.contains('beta')] # less problematic 

## mean % error 

# is error contained in certain regions? 
d['errorMPF'] = np.abs(d['Actual']-d['MPF'])
d['errorLogistic'] = np.abs(d['Actual']-d['transform'])

d.groupby('n_questions')['errorMPF', 'errorLogistic'].mean() # not clear relationship
d.groupby('true_beta')['errorMPF', 'errorLogistic'].mean() # maybe tendency but unclear 

# looking at states instead 
fig, ax = plt.subplots()
for n_questions, true_beta in zip(["3", "5", "7"], ["0.25", "0.5", "1.0"]):

    x = d[(d['n_questions'] == n_questions) & (d['true_beta'] == true_beta)]
    hi_Logistic = x[x['param'].str.contains('alpha')]['transform'].values
    Jij_Logistic = x[x['param'].str.contains('beta')]['transform'].values

    hi_True = x[x['param'].str.contains('alpha')]['Actual'].values
    Jij_True = x[x['param'].str.contains('beta')]['Actual'].values

    hi_MPF = x[x['param'].str.contains('alpha')]['MPF'].values
    Jij_MPF = x[x['param'].str.contains('beta')]['MPF'].values

    from fun import p_dist 
    p_Logistic = p_dist(hi_Logistic, Jij_Logistic)
    p_True = p_dist(hi_True, Jij_True)
    p_MPF = p_dist(hi_MPF, Jij_True)

    plt.scatter(p_True, p_MPF, color = 'tab:blue', s = 5)
    plt.scatter(p_True, p_Logistic, color = 'tab:orange', s = 5)

plt.xlabel('True p(config)')
plt.ylabel('Inferred p(config)')
ax.set_yscale('log')
ax.set_xscale('log')
plt.legend(custom_lines, ['MPF', 'Logistic'])
plt.plot([0, 0.5], [0, 0.5], 'k-')