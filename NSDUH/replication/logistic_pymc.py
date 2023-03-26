import numpy as np 
import pymc as pm 
import pandas as pd 
import pytensor.tensor as at
import arviz as az 

# load data
d = pd.read_csv('../data/replication/jones_nock_2022.csv')

# recode -1 to 0 
# does appear to actually help 
d = d.replace(-1, 0)

# consider major depression (lifetime) first
d_life = d.drop(columns = {'AMDEYR'})

## run just for some values of our 
d_sub = d_life[d_life['IRSEX'] == 1]
data_sub = d_sub[d_sub['CATAG6'] == 3]
data_sub = data_sub[data_sub['NEWRACE2'] == 1]

# now get rid of demographics & filedate
data_sub = data_sub.drop(columns = ['IRSEX', 'CATAG6',
                                    'NEWRACE2', 'IRMARIT'])

# fit it 
d_sub_pred = data_sub.drop(columns = {'AMDELT'})
d_sub_out = data_sub['AMDELT']

model = pm.Model(coords={"predictors": d_sub_pred.columns.values})
with model: 
    alpha = pm.Normal("alpha", mu=0, sigma=5)
    beta = pm.Normal("beta", mu=0, sigma=5, dims="predictors")
    p = pm.Deterministic("p", pm.math.invlogit(alpha + at.dot(d_sub_pred.values, beta)))
    outcome = pm.Bernoulli("outcome", p, observed=d_sub_out.values)
with model: 
    idata = pm.sample(return_inferencedata=True)

az.summary(idata, var_names=['alpha', 'beta'], round_to=2)
idata.to_netcdf("../data/replication/IRSEX1.CATAG63.NEWRACE21.nc")