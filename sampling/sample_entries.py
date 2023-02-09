import pandas as pd 
import numpy as np 
pd.set_option('display.max_colwidth', None)
import random, string
from sample_fun import save_dat, randomword

# ./mpf -l test.dat 1.0 1

# conversion dictionary to convert from 0, 1, -1 to 0, 1, X
conversion_dict = {
    '-1': '0',
    '0': 'X',
    '1': '1'
}

# load data
full_data = pd.read_csv('../data/reference/direct_reference_questions_20_maxna_5_nrows_455_entries_407.csv')
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')
question_ids = question_reference['question_id_drh'].tolist()
question_ids = [str(x) for x in question_ids]

# take out random observations (entry_ids, not rows)
frac = 0.5
unique_entry = full_data[['entry_id']].drop_duplicates()
subsample_entry = unique_entry.sample(frac=frac)
subsample_entry = subsample_entry.sort_values(by='entry_id')
subsample_data = pd.merge(subsample_entry, full_data, on='entry_id', how='inner')

# create corresponding array 
subsample_array = subsample_data[question_ids].to_numpy()

# (1.2) 
weight_string = subsample_data['weight'].astype(str).tolist() 
bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in subsample_array]

# (1.3) save stuff 
id = randomword(10) 
save_dat(bit_string, weight_string, subsample_array, 
         f'../data/subsample_entries/clean/q_20_nan_5_{frac}_{id}.dat')
subsample_data.to_csv(f'../data/subsample_entries/reference/q_20_nan_5_{frac}_{id}.csv', index = False)