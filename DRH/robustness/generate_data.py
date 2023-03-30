import pandas as pd 
import numpy as np 

# load data 
data_raw = pd.read_csv('../data/raw/drh_20221019.csv')
entry_reference = pd.read_csv('../data/analysis/entry_reference.csv')

# for each entry id, get 
data_year = data_raw[['entry_id', 'start_year', 'end_year']]
data_year = data_year.groupby('entry_id').agg({'start_year': 'min', 'end_year': 'max'}).reset_index()

# check whether we can actually reasonably do this 
protestant_reformation = 1517
protestant_reformation = data_year[data_year['start_year'] < protestant_reformation]

# select questions 
question_reference = pd.read_csv('../data/analysis/question_reference.csv')
question_reference = question_reference[['question_id_drh', 'question']]
question_reference
# select only what we have in luther
original_data = pd.read_csv('../data/reference/direct_reference_questions_20_maxna_5_nrows_455_entries_407.csv')
pre_reformation = original_data[original_data['entry_id'].isin(protestant_reformation['entry_id'])] # less than half 

# try with a set of attributes
featues = [
    4676, # official political support
    4729, # scriptures
    4745, # monuments
    4776, # spirit-body distinction
    4787, # reincarnation in this world
    4814, # grave goods
    4821, # formal burials
    4954, # monitoring
    5152, # small-scale rituals
]

# subset both the original data & the pre-reformation data
columns = ['entry_id'] + [str(x) for x in featues] + ['weight']
original_data = original_data[columns]
pre_reformation = pre_reformation[columns]

# put it in the MPF format and save 
def data_to_mpf(d, conversion_dict, filename): 
    d = d.sample(frac=1).reset_index(drop=True)
    d.to_csv(f'../data/robustness/time_reference/{filename}.csv', index=False)
    w = d['weight'].tolist()
    d = d.drop(columns=['entry_id', 'weight'])
    A = d.to_numpy()
    bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in A]
    rows, cols = A.shape
    #w = [str(x) for x in range(cols)]
    with open(f'../data/robustness/time_mpf/{filename}.txt', 'w') as f:
        f.write(f'{rows}\n{cols}\n')
        for bit, weight in zip(bit_string, w): 
            f.write(f'{bit} {weight}\n')

conversion_dict = {
    '-1': '0',
    '0': 'X',
    '1': '1'
}

data_to_mpf(original_data, conversion_dict, 'full_data')
data_to_mpf(pre_reformation, conversion_dict, 'pre_reformation')

'''
x = x.mean().reset_index(name='x_mean')
y = y.mean().reset_index(name='y_mean')
z = x.merge(y, on = 'index', how = 'inner')
z = z[(z['index'] != 'entry_id') & (z['index'] != 'weight')]
z.rename(columns = {'index': 'question_id_drh'}, inplace = True)
z['question_id_drh'] = z['question_id_drh'].astype(int)

question_differences = z.merge(question_reference, on = 'question_id_drh', how = 'inner')
question_differences['difference'] = question_differences['x_mean'] - question_differences['y_mean']
question_differences['abs_difference'] = question_differences['difference'].abs()
question_differences.sort_values('abs_difference', ascending=False)

# inner join with the data we already have
entry_reference = entry_reference[['entry_id', 'entry_name']]
#entry_reference = entry_reference.rename(columns = {'entry_id_drh': 'entry_id'})
year_reference = entry_reference.merge(data_year, on = 'entry_id', how = 'inner')

# sort by start year
year_reference = year_reference.sort_values('start_year', ascending=True)
'''
