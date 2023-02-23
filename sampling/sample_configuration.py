import pandas as pd 
import numpy as np 
pd.set_option('display.max_colwidth', None)
import random, string
from sample_fun import save_dat, randomword
from tqdm import tqdm 
import itertools

# conversion dictionary to convert from 0, 1, -1 to 0, 1, X
conversion_dict = {
    -1: '0',
    0: 'X',
    1: '1'
}

def remove_zero_rows(df, columns):
    mask = (df[columns] != 0).any(axis=1)
    return df[mask]
 
# load data
full_data = pd.read_csv('../data/reference/direct_reference_questions_20_maxna_5_nrows_455_entries_407.csv')
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')
question_ids = question_reference['question_id_drh'].tolist()
question_ids = [str(x) for x in question_ids]

# sequentially add new nodes to the same basic system 
num_questions = 3
init_system = random.sample(question_ids, num_questions - 1)
system_list = []
for i in range(8): # up to n=10 
    other_questions = [x for x in question_ids if x not in init_system]
    init_system = init_system + random.sample(other_questions, 1)
    system_list.append(init_system)

#questions = system_list[0]
#sorted_questions = sorted(questions)
#sorted_columns = ['entry_id'] + sorted_questions + ['weight']
#sorted_data = full_data[sorted_columns]
#weighted_data = sorted_data.groupby(['entry_id'] + sorted_questions)['weight'].sum()

for questions in system_list: 
    sorted_questions = questions
    sorted_columns = ['entry_id'] + sorted_questions + ['weight']
    sorted_data = full_data[sorted_columns]

    # collapse entries that now have the same questions 
    weighted_data = sorted_data.groupby(['entry_id'] + sorted_questions)['weight'].sum().reset_index(name = 'weight')
    cleaned_data = remove_zero_rows(weighted_data, sorted_questions)
    cleaned_array = cleaned_data[sorted_questions].to_numpy()
    
    weight_string = cleaned_data['weight'].astype(str).tolist()
    bit_string = ["".join([conversion_dict[x] for x in row]) for row in cleaned_array]

    # save data 
    rows, cols = cleaned_array.shape
    id = randomword(10) 
    save_dat(bit_string, weight_string, cleaned_array, 
            f'../data/sample_questions/mdl_config/q20_nan5_q{cols}_id{id}.dat')
    cleaned_data.to_csv(f'../data/sample_questions/reference_config/q20_nan5_q{cols}_id{id}.csv', index = False)
