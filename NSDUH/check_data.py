import pandas as pd 
import numpy as np 

# read data
d = pd.read_csv('NSDUH_2020_Tab.txt', sep = "\t")
len(d) # 32893 rows; 2890 columns
d.groupby('FILEDATE').size() # this is only 2020/2021 (get more data).

# (1) only adults (18+)
## age in [26-34]
d = d[(d['AGE2'] == 13) | (d['AGE2'] == 14)] # see p. 567 of the codebook
## sex == male


# variables we need: 
''' drugs (this is all lifetime in theirs)
ecstacy, psilocybin, lsd, peyote, mescaline, cocaine, heroin, pcp, inhalant
pain reliever, tranquilizer, stimulant, sedative, marijuana, (alcohol, tobacco).
'''

''' demographics (covariates)
sex, age, race, household income, education, marital status, self-reported risky 
behavior
'''

''' mental health (outcome)
major depression (severe?), serious psychological distress, suicidal ideation,
(anxiety?) 
'''

## problem with the whole recency thing is sparsity .... 
# so; for e.g. MDMA we have either lifetime which they use (8.14%)
# or we need to recode the "within last 30 days" (0.3%)
# or within last year (1.25%)

# list of drug month/year variables 
recencylist = [
    # MOLLY/HALLUCINOGENS
    'ECSTMOREC', # ecstacy/mdma (1=past 30, 2=past 12mo)
    # PSILOCYBIN only exists as EVER/LIFETIME
    # MESCALIN only exists as EVER/LIFETIME
    # PEYOTE only exists as EVER/LIFETIME
    'PCPREC', # pcp (1=past 30, 2=past 12mo)
    'LSDREC', # lsd (1=past 30, 2=past 12mo)
    
    ## illicit drugs 
    'COCREC', # cocaine (1=past 30, 2=past 12mo)
    'HERREC', # heroin (1=past 30, 2=past 12mo)
    'MJREC', # marijuana (1=past 30, 2=past 12mo)
    
    ## legal drugs 
    'PNRANYREC', # pain reliever (1=past 30, 2=past 12mo)
    'INHALREC' # inhalant (1=past 30, 2=past 12mo)
    'STMANYREC', # stimulant (1=past 30, 2=past 12mo)
    'SEDANYREC', # sedative (1=past 30, 2=past 12mo)
]

# list of drug ever/lifetime variables
everlist = [
    ## MOLLY/HALLUCINOGENS 
    'ECSTMOLLY', # ever ecstacy/mdma
    'PSILCY', # ever psilocybin
    'MESC', # ever mescaline
    'PEYOTE', # ever peyote
    'PCP', # ever pcp
    'LSD', # ever lsd
    
    ## illicit drugs 
    'COCEVER', # ever cocaine
    'HEREVER', # ever heroin
    'MJEVER', # ever marijuana
    
    ## legal drugs 
    'PNRANYLIF', # ever pain reliever
    'INHALEVER', # ever inhalant
    'STMANYLIF', # ever stimulant
    'SEDANYLIF', # ever sedative
    ]

# list of mental health variables
# probably want to do lifetime (although the papers typically try all of these as outcomes).
mental_list = [
    #'AMDELT', # major depression (lifetime) 
    'AMDEYR', # major depression (last year)
    'SPDYR', # serious psychological distress (last year)
    #'SPDMON', # serious psychological distress (last month)
    'MHSUITHK', # suicidal ideation (last year)
    'MHSUIPLN', # suicidal planning (last year)
    'MHSUITRY', # suicidal attempt (last year)
]

## try to see how much data we have ## 
# (1) subset
mental_ever = everlist + mental_list 
d_mental_ever = d[mental_ever]
# (2) recode
recode_general = {
    0: 0, # unknown (how we recode it) 
    1: 1, # Yes
    2: -1, # No
    3: 1, # Yes logically assigned
    5: 1, # Yes logically assigned 
    85: 0, # Bad data logically assigned
    91: -1, # No (never used super-category)
    94: 0, # don't know
    97: 0, # refused to report 
    98: 0 # blank no answer 
          }
recode_float = {
    0.0: 2, # No
    1.0: 1, # Yes,
    np.nan: 0 # unknown
    }

# why do we have nan here when only above 18?
float_columns = ['AMDEYR', 'SPDYR', 'MHSUITHK', 'MHSUIPLN', 'MHSUITRY']
d_mental_ever[float_columns] = d_mental_ever[float_columns].replace(recode_float).astype(int)
d_mental_ever = d_mental_ever.replace(recode_general)

# how many rows with LEQ 5 NAN
# Count the number of zeros in each row
zero_counts = (d_mental_ever == 0).sum(axis=1)
d_mental_LEQ5 = d_mental_ever[zero_counts <= 5]
d_mental_LEQ5 # almost all of them remain 

# how good coverage do the questions have?
d_mental_ever.mean(axis=0) # most strongly negative 
