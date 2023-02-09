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

# subsample questions 
n_questions = 10
subsample_questions = random.sample(question_ids, n_questions)
subsample_questions = sorted(subsample_questions)

subsample_columns = ['entry_id'] + subsample_questions + ['weight']
subsample_data = full_data[subsample_columns]

# create corresponding array 
subsample_array = subsample_data[subsample_questions].to_numpy()

# wrangle data to be in the right format 
weight_string = subsample_data['weight'].astype(str).tolist() 
bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in subsample_array]

# save data 
id = randomword(10) 
save_dat(bit_string, weight_string, subsample_array, 
         f'../data/sample_questions/mdl_input/q20_nan5_qsub{n_questions}_id{id}.dat')
subsample_data.to_csv(f'../data/sample_questions/reference/q20_nan5_qsub{n_questions}_id{id}.csv', index = False)