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

## Q: some of these are likert (e.g. education)
## some are qualitative; how to treat them? 

# for now only columns without nan 
d_no_nan = d.loc[d.apply(lambda row: 0 not in row.values, axis=1)]
d_no_nan # 167.585 rows 

# drop the year column
d_no_nan = d.drop(columns = ['year'])

# save data 
d_no_nan.to_csv('../data/replication/jones_nock_2022.csv', index = False)