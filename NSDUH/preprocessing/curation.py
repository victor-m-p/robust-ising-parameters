'''
VMP 2023-03-17:
downloaded files from: https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health
here we preprocess all files from 2008-2019 and 
write a file ready for processing with .MPF 
'''

# loads 
import re
import pandas as pd 
import numpy as np 
import os 

# columns that we care about 
## select only the columns we care about 
columns_2016_2019 = [
    # drugs 
    'CIGREC',
    'ALCREC',
    'MJREC',
    'COCREC',
    'HERREC',
    'LSDREC',
    'PCPREC',
    'ECSTMOREC',
    'INHALREC',
    'PNRNMREC',
    'TRQNMREC',
    'STMNMREC',
    'SEDNMREC',
    # demographics
    #'AGE2',
    'IRSEX',
    'IRMARIT',
    'CATAG6',
    'NEWRACE2',
    'IREDUHIGHST2', # education
    'INCOME',
    # mental health / depression
    'SUICTHNK',
    'SUICPLAN',
    'SUICTRY',
    'AMDEYR',
    'SPDYR'
]

## conversion from 2016-2019 to 2008-2014
conversion_new_to_old = {
    'ECSTMOREC': 'ECSREC',
    'INHALREC': 'INHREC',
    'PNRNMREC': 'ANALREC',
    'TRQNMREC': 'TRANREC',
    'STMNMREC': 'STIMREC',
    'SEDNMREC': 'SEDREC',
    'IREDUHIGHST2': 'IREDUC2'
}

columns_2008_2014 = [conversion_new_to_old.get(x, x) for x in columns_2016_2019]

## conversion from 2016-2019 to 2015
conversion_new_to_2015 = {
    'IRMARIT': 'IRMARITSTAT'
}
columns_2015 = [conversion_new_to_2015.get(x, x) for x in columns_2016_2019]

# read files 
dir_path = '../data/raw/'
files = [file for file in os.listdir(dir_path) if file.endswith(('.tsv', '.txt'))]
subfiles = [f for f in files if int(re.match('NSDUH_(\d{4})', f).group(1)) <= 2019]

data_2016_2019 = []
data_2008_2014 = []
data_2015 = []
for f in subfiles:
    print(f)
    d = pd.read_csv(dir_path + f, sep = "\t") 
    year = int(re.match('NSDUH_(\d{4})', f).group(1))
    if year <= 2014:
        d = d[columns_2008_2014]
        d['year'] = year
        data_2008_2014.append(d) 
    elif year == 2015:
        d = d[columns_2015]
        d['year'] = year
        data_2015.append(d)
    else: 
        d = d[columns_2016_2019]
        d['year'] = year
        data_2016_2019.append(d) 

# create the dataframes 
data_2016_2019 = pd.concat(data_2016_2019)
data_2015 = pd.concat(data_2015)
data_2008_2014 = pd.concat(data_2008_2014)

# the variables appear to be consistently coded
# across time, including the ones that differ in
# name, so here we rename the columns to the 
# 2016-2019 format 
data_2015 = data_2015.rename(
    columns = {value: key for key, value in conversion_new_to_2015.items()}
)
data_2008_2014 = data_2008_2014.rename(
    columns = {value: key for key, value in conversion_new_to_old.items()}
)
data_all_years = pd.concat([data_2016_2019, data_2015, data_2008_2014])

# recode the files to 1 (yes) / -1 (no) / 0 (don't know)
## drugs 
### CIGREC, ALCREC, MJEREC, COCREC, HERREC, LSDREC
### PCPREC, ECSTMOREC, INHALREC, 
### PNRNMREC, TRQNMREC,STMNMREC, SEDREC
recode_drugs = {
    1: 1, # Yes (30 days)
    2: 1, # Yes (30 days - 12 month)
    3: -1, # No (+12 months ago)
    4: -1, # No (+3 years ago)
    8: 1, # Yes (12 month logic-assign)
    9: -1, # No (some point in life logic-assign)
    11: 1, # Yes (used in past 30 days logic-assign)
    12: 1, # Yes (+30 days but within 12mo logic-assign)
    14: -1, # No (+12 months ago logic-assign)
    19: -1, # No (More than 30 days ago logic-assign)
    29: -1, # No (More than 30 days ago but in past 3yr logic-assign)
    39: -1, # No (with 3 years logic-assign)
    81: -1, # No (never used logic-assign)
    83: -1, # No (no misuse past 12 mo (lifetime?) logic-assign)
    85: 0, # D/K (BAD DATA logic-assign)
    91: -1, # No (never used) 
    97: 0, # D/K (refused to answer)
    98: 0, # D/K (no answer)
    }

drug_cols = ['CIGREC', 'ALCREC', 'MJREC', 'COCREC', 'HERREC',
             'LSDREC', 'PCPREC', 'ECSTMOREC', 'INHALREC',
             'PNRNMREC', 'TRQNMREC', 'STMNMREC', 'SEDNMREC']

data_all_years[drug_cols] = data_all_years[drug_cols].replace(recode_drugs)

## suicide 
### SUICTHNK, SUICPLAN, SUICTRY
recode_suicide = {
    1: 1, # Yes
    2: -1, # No
    85: 0, # D/K (BAD DATA logic-assign)
    89: 0, # D/K (legitimate skip logic-assign)
    94: 0, # D/K (don't know)
    97: 0, # D/K (refused to answer)
    98: 0, # D/K (no answer)
    99: 0, # D/K (legitimate skip)
}

suicide_cols = ['SUICTHNK', 'SUICPLAN', 'SUICTRY']
data_all_years[suicide_cols] = data_all_years[suicide_cols].replace(recode_suicide)

## major depression (AMDEYR)
recode_depression = {
    1.0: 1, # Yes
    2.0: -1, # No
    np.nan: 0 # D/K (unknown or youth- but we filter those)
    }
data_all_years['AMDEYR'] = data_all_years['AMDEYR'].replace(recode_depression).astype(int)

## serious psychological distress (SPDYR)
recode_distress = {
    0.0: -1, # No
    1.0: 1, # Yes
    np.nan: 0, # D/K (unknown or youth- but we filter those)
}
data_all_years['SPDYR'] = data_all_years['SPDYR'].replace(recode_distress).astype(int)

# recode demographics 
def split_likert_column(d, column): 
    d_binary = d.groupby(column).size().reset_index(name='count')
    d_binary = d_binary.sort_values(by = column)
    d_binary['cumsum'] = d_binary['count'].cumsum()
    # Calculate the total sum
    total_sum = d_binary['count'].sum()
    # Find the split point that minimizes the difference between the sums of the two groups
    split_point = (d_binary['cumsum'] - total_sum / 2).abs().idxmin()
    d_binary[f'{column}_binary'] = d_binary.index.to_series().apply(lambda x: -1 if x <= split_point else 1)
    d_binary = d_binary[[column, f'{column}_binary']]
    d_binary = d.merge(d_binary, on = column, how = 'inner')
    return d_binary

data_all_years = split_likert_column(data_all_years, 'IREDUHIGHST2') 
data_all_years = split_likert_column(data_all_years, 'CATAG6')
data_all_years = split_likert_column(data_all_years, 'INCOME')

## check splits 
### edata_all_yearsucation: splits between high-school and some college
data_all_years.groupby(['IREDUHIGHST2', 'IREDUHIGHST2_binary']).size().reset_index(name='count')
### age: splits [18:34] vs [35+] 
data_all_years.groupby(['CATAG6', 'CATAG6_binary']).size().reset_index(name='count')
### income: splits [<49.999] vs [>=50.000]
data_all_years.groupby(['INCOME', 'INCOME_binary']).size().reset_index(name='count')

## non-likert columns split manually 
### gendata_all_yearser alreadata_all_yearsy data_all_yearsone 
data_all_years['IRSEX_binary'] = data_all_years['IRSEX'].apply(lambda x: -1 if x == 1 else 1)
### race (white vs. non-white) with white as reference
data_all_years['NEWRACE2_binary'] = data_all_years['NEWRACE2'].apply(lambda x: -1 if x == 1 else 1)
### marital status: currently married as reference. 
data_all_years['IRMARIT_binary'] = data_all_years['IRMARIT'].apply(lambda x: -1 if x == 1 else 1)

## now remove columns that we do not use 
data_all_years = data_all_years.drop(columns = ['IRSEX', 'IRMARIT', 'IREDUHIGHST2', 'CATAG6', 'INCOME', 'NEWRACE2', 'year'])

# filter demographics 
### IRSEX: 1 = Male, 2 = Female 
### IRMARIT/IRMARITSTAT: 1 = Married, ..., 4 = Never
### CATAGE: 3 = 26-34, ...
### NEWRACE2: 1 = White (non-hispanic), ...
#data_all_years.to_csv('../data/reference/NSDUH_pre_demographics.csv', index=False)
#len(data_all_years) # 674.521
#data_subset = data_all_years[data_all_years['IRSEX'] == 1]
#len(data_subset) # 322.636
#data_subset = data_subset[data_subset['CATAGE'] == 3]
#len(data_subset) # 39.472
#data_subset = data_subset[data_subset['NEWRACE2'] == 1]
#len(data_subset) # 23.516

# now get rid of demographics & filedate
#data_subset = data_subset.drop(columns = ['IRSEX', 'CATAGE', 
#                                          'NEWRACE2', 'IRMARIT', 
#                                          'SUICPLAN', 'SUICTRY',
#                                          'year'])

# remove the two crazy columns 
zero_counts = (data_all_years == 0).sum(axis=1)
d_LEQ5 = data_all_years[zero_counts <= 5]

# put it in the MPF format and save 
def data_to_mpf(d, conversion_dict, filename): 
    d = d.sample(frac=1).reset_index(drop=True)
    d.to_csv(f'../data/reference/{filename}.csv', index=False)
    A = d.to_numpy()
    bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in A]
    rows, cols = A.shape
    w = [str(1.0) for _ in range(cols)]
    with open(f'../data/clean_splits/{filename}.txt', 'w') as f:
        f.write(f'{rows}\n{cols}\n')
        for bit, weight in zip(bit_string, w): 
            f.write(f'{bit} {weight}\n')

conversion_dict = {
    '-1': '0',
    '0': 'X',
    '1': '1'
}

data_to_mpf(d_LEQ5, conversion_dict, 'NSDUH_full')

# all combinations of the columns we want to vary 
import itertools 
lst = list(itertools.product([-1, 1], repeat=6))
variables = ['IREDUHIGHST2_binary', 'CATAG6_binary', 'INCOME_binary', 'IRSEX_binary', 'IRMARIT_binary', 'NEWRACE2_binary']

for demographic_split in lst:
    i = list(demographic_split)
    dsub = d_LEQ5[(d_LEQ5[variables[0]] == i[0]) & 
                  (d_LEQ5[variables[1]] == i[1]) & 
                  (d_LEQ5[variables[2]] == i[2]) & 
                  (d_LEQ5[variables[3]] == i[3]) & 
                  (d_LEQ5[variables[4]] == i[4]) & 
                  (d_LEQ5[variables[5]] == i[5])].drop(columns = variables)
    data_to_mpf(dsub, conversion_dict, f'NSDUH_full_{i[0]}_{i[1]}_{i[2]}_{i[3]}_{i[4]}_{i[5]}')
    