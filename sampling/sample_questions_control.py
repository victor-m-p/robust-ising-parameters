import pandas as pd 
import numpy as np 
pd.set_option('display.max_colwidth', None)
import random, string
from sample_fun import save_dat, randomword
from tqdm import tqdm 
import itertools

# conversion dictionary to convert from 0, 1, -1 to 0, 1, X
conversion_dict = {
    '-1': '0',
    '0': 'X',
    '1': '1'
}

def remove_zero_rows(df, columns):
    mask = (df[columns] != 0).any(axis=1)
    return df[mask]
 
# load data
full_data = pd.read_csv('../data/reference/direct_reference_questions_20_maxna_5_nrows_455_entries_407.csv')
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')
question_ids = question_reference['question_id_drh'].tolist()
question_ids = [str(x) for x in question_ids]

# all two-question combinations
num_questions = 5
combinations = list(itertools.combinations(question_ids, 2))
five_random = random.sample(combinations, 10) # for now 

# for now match the other (5 rather than 10) 
five_random = [
    ("4676", "5152"),
    ("4745", "5137"),
    ("4776", "4808"),
    ("4776", "4827"),
    ("4780", "5220")
]

for i, j in five_random:
    focus_Jij = [i, j]
    other_questions = [x for x in question_ids if x not in focus_Jij]
    
    for sample in range(100): # should this range change is the question

        # subsample questions  
        subsample_questions = random.sample(other_questions, num_questions - 2)
        subsample_questions = sorted(subsample_questions)
        subsample_questions = focus_Jij + subsample_questions
        subsample_columns = ['entry_id'] + subsample_questions + ['weight']
        subsample_data = full_data[subsample_columns]

        # collapse entries that now have the same questions 
        subsample_data = subsample_data.groupby(['entry_id'] + subsample_questions)['weight'].sum().reset_index(name = 'weight')

        # remove potential nan rows 
        subsample_data = remove_zero_rows(subsample_data, subsample_questions)

        # create corresponding array 
        subsample_array = subsample_data[subsample_questions].to_numpy()

        # wrangle data to be in the right format 
        weight_string = subsample_data['weight'].astype(str).tolist() 
        bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in subsample_array]

        # save data 
        id = randomword(10) 
        save_dat(bit_string, weight_string, subsample_array, 
                f'../data/sample_questions/mdl_10/q20_nan5_n{num_questions}_i{i}_j{j}_sample{sample}_id{id}.dat')
        subsample_data.to_csv(f'../data/sample_questions/reference_10/q20_nan5_n{num_questions}_i{i}_j{j}_sample{sample}_id{id}.csv', index = False)