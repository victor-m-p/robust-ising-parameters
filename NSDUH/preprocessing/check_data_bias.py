import pandas as pd 
import numpy as np 

d = pd.read_csv('../data/reference/NSDUH_pre_demographics.csv')
d = d.drop(columns=['SUICTRY', 'SUICPLAN'])

# volume over time: 
d.groupby('year').count() # even

# % over time 
mean_grouped = d.groupby('year').mean()

# plot each of them over time 
# basically looks fine 