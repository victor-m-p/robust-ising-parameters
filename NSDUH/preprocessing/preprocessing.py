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

d = pd.read_csv('../data/reference/NSDUH_2008_2019_NAN5.csv')
d

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
    'CATAGE',
    'NEWRACE2',
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
    'SEDNMREC': 'SEDREC'
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
    if int(re.match('NSDUH_(\d{4})', f).group(1)) <= 2014:
        d = d[columns_2008_2014]
        data_2008_2014.append(d) 
    elif int(re.match('NSDUH_(\d{4})', f).group(1)) == 2015:
        d = d[columns_2015]
        data_2015.append(d)
    else: 
        d = d[columns_2016_2019]
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

# filter demographics 
### IRSEX: 1 = Male, 2 = Female 
### IRMARIT/IRMARITSTAT: 1 = Married, ..., 4 = Never
### CATAGE: 3 = 26-34, ...
### NEWRACE2: 1 = White (non-hispanic), ...
len(data_all_years) # 674.521
data_subset = data_all_years[data_all_years['IRSEX'] == 1]
len(data_subset) # 322.636
data_subset = data_subset[data_subset['CATAGE'] == 3]
len(data_subset) # 39.472
data_subset = data_subset[data_subset['NEWRACE2'] == 1]
len(data_subset) # 23.516

# now get rid of demographics
data_subset = data_subset.drop(columns = ['IRSEX', 'CATAGE', 'NEWRACE2', 'IRMARIT'])

# only the ones that have less than 5 nan 
zero_counts = (data_subset == 0).sum(axis=1)
d_LEQ5 = data_subset[zero_counts <= 5]

# save to reference 
d_LEQ5.to_csv('../data/reference/NSDUH_2008_2019_NAN5.csv', index=False)

# put it in the MPF format and save 
A_LEQ5 = d_LEQ5.to_numpy()
conversion_dict = {
    '-1': '0',
    '0': 'X',
    '1': '1'
}
bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in A_LEQ5]
w = np.ones(len(A_LEQ5))
rows, cols = A_LEQ5.shape
with open('../data/clean/NSDUH_2008_2019_NAN5.txt', 'w') as f: 
    f.write(f'{rows}\n{cols}\n')
    for bit, weight in zip(bit_string, w): 
        f.write(f'{bit} {weight}\n')
        
# remove the two crazy columns 
data_subset = data_subset.drop(columns = ['SUICPLAN', 'SUICTRY'])
zero_counts = (data_subset == 0).sum(axis=1)
d_LEQ5.to_csv('../data/reference/NSDUH_2008_2019_NAN5_SUIC.csv', index=False)

# put it in the MPF format and save 
A_LEQ5 = d_LEQ5.to_numpy()
conversion_dict = {
    '-1': '0',
    '0': 'X',
    '1': '1'
}
bit_string = ["".join(conversion_dict.get(str(int(x))) for x in row) for row in A_LEQ5]
w = np.ones(len(A_LEQ5))
rows, cols = A_LEQ5.shape
with open('../data/clean/NSDUH_2008_2019_NAN5_SUIC.txt', 'w') as f: 
    f.write(f'{rows}\n{cols}\n')
    for bit, weight in zip(bit_string, w): 
        f.write(f'{bit} {weight}\n')
 