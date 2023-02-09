import pandas as pd 
import numpy as np 
import os 
import re
from collections import defaultdict
import itertools 
import matplotlib.pyplot as plt 
import seaborn as sns 

files = sorted(os.listdir('../data/sample_entries/reference'))
d = pd.read_csv(f'../data/sample_entries/reference/{files[0]}')
 
# get the entries for each sample
dict_sample_entries = defaultdict(dict)
for file in files: 
    number_sample = int(re.search(r'n(\d+)_id', file)[1])
    number_entries = int(re.search(r'entries(\d+)_f', file)[1])
    d = pd.read_csv(f'../data/sample_entries/reference/{file}')
    entries = d['entry_id'].unique().tolist()
    dict_sample_entries[number_entries][number_sample] = entries

# check Jij and hi
number_entries = sorted([x for x in dict_sample_entries.keys()])

average_num_hi = 10
unique_entry = 407

dict_Jij = defaultdict(int)
dict_hi = defaultdict(int)
dict_Jij_super = defaultdict(dict)
dict_hi_super = defaultdict(dict)
for n_entries in sorted([x for x in dict_sample_entries.keys()]):
    dict_number_entries = defaultdict(int)
    dict_number_hi = defaultdict(int)
    num_hi = int(round(average_num_hi * unique_entry/n_entries)) 
    for sample in range(num_hi): 
        entries = dict_sample_entries[n_entries][sample]
        for comb in itertools.combinations(entries, 2):
            dict_number_entries[comb] += 1
        for question in entries: 
            dict_number_hi[question] += 1
    dict_Jij_super[n_entries] = dict_number_entries
    dict_hi_super[n_entries] = dict_number_hi

dict_hi_super

# check Jij
number_entries
Jij_count_super = []
for n_entries in [41, 204, 366]:
    sub_dict = dict_Jij_super[n_entries]
    Jij_count_list = []
    for v in sub_dict.values(): 
        Jij_count_list.append((n_entries, v))
    Jij_counts = pd.DataFrame(Jij_count_list, columns = ['n_entries', 'count'])
    Jij_count_super.append(Jij_counts)
Jij_counts_df = pd.concat(Jij_count_super)

# plot 
fig, ax = plt.subplots(figsize = (10, 10), dpi = 300)
sns.kdeplot(data = Jij_counts_df, x = 'count', 
            hue = 'n_entries', fill = True, 
            bw_method = 0.5, ax = ax)
plt.savefig('../figures/sample_entries_Jij_check.pdf', bbox_inches = 'tight')

# average number of connections per Jij
Jij_counts_df.groupby('n_entries')['count'].mean() 

# check hi
hi_count_super = []
for n_entries in [41, 204, 366]:
    sub_dict = dict_hi_super[n_entries]
    hi_count_list = []
    for v in sub_dict.values():
        hi_count_list.append((n_entries, v))
    hi_counts = pd.DataFrame(hi_count_list, columns = ['n_entries', 'count'])
    hi_count_super.append(hi_counts)
hi_counts_df = pd.concat(hi_count_super)

# plot
fig, ax = plt.subplots(figsize = (10, 10), dpi = 300)
sns.kdeplot(data = hi_counts_df, x = 'count',
            hue = 'n_entries', fill = True,
            bw_method = 0.5, ax = ax)
plt.savefig('../figures/sample_entries_hi_check.pdf', bbox_inches = 'tight')

hi_counts_df.head(5)