'''
VMP 2023-03-25:
downloaded files from: https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health
Trying to reproduce (roughly) the analysis reported in Jones, Nick (2022).
Currently not including 2014 because variables change (not sure why they did this).
Much more natural to use 2015-2019 for instance ...
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
    'ECSTMOLLY',
    'PSILCY',
    'LSD',
    'PEYOTE',
    'MESC',
    'COCEVER',
    'HEREVER',
    'PCP',
    'INHALEVER',
    'PNRANYLIF',
    'TRQANYLIF',
    'STMANYLIF',
    'SEDANYLIF',
    'MJEVER',
    # demographics
    #'AGE2',
    'IRSEX', # probably the same, but check levels
    'IRMARIT', # probably the same, but check levels
    'CATAG6', # probably the same, but check levels
    'NEWRACE2', # probable the same, but check levels
    'IREDUHIGHST2', # education
    'INCOME', # family income
    'RSKYFQTES', # risky behavior
    # mental health / depression
    'AMDELT', # lifetime major depression
    'AMDEYR' # last year major depression
    ## not sure about the "severe" depression last year ...
]

## conversion from 2016-2019 to 2015
conversion_new_to_2015 = {
    'IRMARIT': 'IRMARITSTAT'
}
columns_2015 = [conversion_new_to_2015.get(x, x) for x in columns_2016_2019]

# read files 
dir_path = '../data/raw/'
files = [file for file in os.listdir(dir_path) if file.endswith(('.tsv', '.txt'))]
subfiles = [f for f in files if 2015 <= int(re.match('NSDUH_(\d{4})', f).group(1)) <= 2018]

data = []
for f in subfiles:
    print(f)
    d = pd.read_csv(dir_path + f, sep = "\t") 
    year = int(re.match('NSDUH_(\d{4})', f).group(1))
    if year == 2015: # IRMARIT column name changed.
        d = d[columns_2015]
        d = d.rename(columns = {'IRMARITSTAT': 'IRMARIT'})
    else: 
        d = d[columns_2016_2019]
    d['year'] = year
    data.append(d) 

# create the dataframes 
d = pd.concat(data) # 226.632

# recode the files to 1 (yes) / -1 (no) / 0 (don't know)
recode_drugs = {
    1: 1, # Yes 
    2: -1, # No 
    3: 1, # Yes (logical)
    5: 1, # Yes (logical)
    85: 0, # Bad data (assign D/K)
    91: -1, # No (never used) 
    94: 0, # D/K 
    97: 0, # D/K (refused to answer)
    98: 0, # D/K (no answer / blank)
    }

drug_cols = ['ECSTMOLLY', 'PSILCY', 'LSD', 'PEYOTE',
             'MESC', 'COCEVER', 'HEREVER', 'PCP',
             'INHALEVER', 'PNRANYLIF', 'TRQANYLIF',
             'STMANYLIF', 'SEDANYLIF', 'MJEVER']

d[drug_cols] = d[drug_cols].replace(recode_drugs)

# major depression
## AMDEYR, AMDELT
recode_depression = {
    1.0: 1, # Yes
    2.0: -1, # No
    np.nan: 0 # D/K (unknown or youth- but we filter those)
    }
d[['AMDEYR', 'AMDELT']] = d[['AMDEYR', 'AMDELT']].replace(recode_depression).astype(int)

# demographics 
## (1) remove children
d['CATAG6'].unique()
d = d[d['CATAG6'] > 1] # 1: (12-17 yo.) 

# coding of demographics...
## IRSEX: 2 levels 
## IRMARIT: 4 levels
## CATAG6: 5 levels 
## NEWRACE2: 7 levels
## IREDUHIGHST2: 11 levels 
## INCOME: 4 levels 
## RSKYFQTES: problematic. 

# recode RSKYFQTES
recode_risky = {
    1: 1,
    2: 2,
    3: 3,
    4: 4, 
    85: 0, # bad data
    94: 0, # don't know
    97: 0, # refused
    98: 0 # blank
    }
d['RSKYFQTES'] = d['RSKYFQTES'].replace(recode_risky)

# binarize demographics 
## likert columns split as evenly as possible 
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

d = split_likert_column(d, 'IREDUHIGHST2') 
d = split_likert_column(d, 'CATAG6')
d = split_likert_column(d, 'INCOME')

## check splits 
### education: splits between high-school and some college
d.groupby(['IREDUHIGHST2', 'IREDUHIGHST2_binary']).size().reset_index(name='count')
### age: splits [18:34] vs [35+] 
d.groupby(['CATAG6', 'CATAG6_binary']).size().reset_index(name='count')
### income: splits [<49.999] vs [>=50.000]
d.groupby(['INCOME', 'INCOME_binary']).size().reset_index(name='count')

## non-likert columns split manually 
### gender already done 
d['IRSEX_binary'] = d['IRSEX'].apply(lambda x: -1 if x == 1 else 1)
### race (white vs. non-white) with white as reference
d['NEWRACE2_binary'] = d['NEWRACE2'].apply(lambda x: -1 if x == 1 else 1)
### marital status: currently married as reference. 
d['IRMARIT_binary'] = d['IRMARIT'].apply(lambda x: -1 if x == 1 else 1)


## now remove columns that we do not use 
d = d.drop(columns = ['IRMARIT', 'IREDUHIGHST2', 'CATAG6', 'INCOME', 'NEWRACE2', 'RSKYFQTES', 'year'])

## Q: some of these are likert (e.g. education)
## some are qualitative; how to treat them? 

# for now only columns without nan 
d_no_nan = d.loc[d.apply(lambda row: 0 not in row.values, axis=1)]

# save data 
d_no_nan.to_csv('../data/replication/jones_nock_2022.csv', index = False)