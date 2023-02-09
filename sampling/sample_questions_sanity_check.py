import pandas as pd 
import numpy as np 
import os 
import re
from collections import defaultdict
import itertools 
import matplotlib.pyplot as plt 
import seaborn as sns 

files = sorted(os.listdir('../data/sample_questions/reference'))

# get the questions for each sample
dict_sample_questions = defaultdict(dict)
for file in files: 
    number_sample = int(re.search(r'n(\d+)_id', file)[1])
    number_questions = int(re.search(r'qsub(\d+)_n', file)[1])
    d = pd.read_csv(f'../data/sample_questions/reference/{file}')
    d = d.drop(columns = ['entry_id', 'weight'])
    questions = d.columns.tolist()
    questions = [int(x) for x in questions]
    dict_sample_questions[number_questions][number_sample] = questions

# this is probably the smartest now 
average_num_connections = 10
max_questions = 20
min_questions = 5
increment_questions = 1
max_Jij = int(max_questions*(max_questions-1)/2)

dict_number_questions = defaultdict(int)
dict_super = defaultdict(dict)
for n_questions in range(5, 20, 1): 
    dict_number_questions = defaultdict(int)
    n_Jij = int(n_questions*(n_questions-1)/2)
    multiplier = 1/(n_Jij/max_Jij)
    num_connections = int(round(multiplier*average_num_connections))
    for sample in range(num_connections): 
        questions = dict_sample_questions[n_questions][sample]
        for comb in itertools.combinations(questions, 2):
            dict_number_questions[comb] += 1
    dict_super[n_questions] = dict_number_questions
    
# test a couple of these 
Jij_count_super = []
for n_questions in [5, 10, 15, 19]:
    sub_dict = dict_super[n_questions]
    Jij_count_list = []
    for k, v in sub_dict.items(): 
        #configuration_x, configuration_y = k
        Jij_count_list.append((n_questions, v))
    Jij_counts = pd.DataFrame(Jij_count_list, columns = ['n_questions', 'count'])
    Jij_count_super.append(Jij_counts)
Jij_counts_df = pd.concat(Jij_count_super)

# plot 
fig, ax = plt.subplots(figsize = (10, 10), dpi = 300)
sns.kdeplot(data = Jij_counts_df, x = 'count', 
            hue = 'n_questions', fill = True, 
            bw_method = 0.5, ax = ax)
plt.savefig('../figures/sample_questions_sanity_check.pdf', bbox_inches = 'tight')

# average number of connections per Jij
Jij_counts_df.groupby('n_questions')['count'].mean() 