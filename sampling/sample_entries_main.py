import pandas as pd 
import numpy as np 
pd.set_option('display.max_colwidth', None)
from sample_fun import save_dat, randomword

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
average_num_hi = 10

for frac in np.arange(0.1, 1, 0.1): 
    num_hi = int(round(average_num_hi * 1/frac)) 
    frac_clean = round(frac,2)
    for sample in range(num_hi):
        unique_entry = full_data[['entry_id']].drop_duplicates()
        num_entries = int(round(len(unique_entry)*frac))
        subsample_entry = unique_entry.sample(n=num_entries)
        subsample_entry = subsample_entry.sort_values(by='entry_id')
        subsample_data = pd.merge(subsample_entry, full_data, on='entry_id', how='inner')    

        subsample_array = subsample_data[question_ids].to_numpy()

        weight_string = subsample_data['weight'].astype(str).tolist() 
        bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in subsample_array]

        id = randomword(10) 
        save_dat(bit_string, weight_string, subsample_array, 
                f'../data/sample_entries/mdl_input/q20_nan5_entries{num_entries}_frac{frac_clean}_n{sample}_id{id}.dat')
        subsample_data.to_csv(f'../data/sample_entries/reference/q20_nan5_entries{num_entries}_frac{frac_clean}_n{sample}_id{id}.csv', index = False)