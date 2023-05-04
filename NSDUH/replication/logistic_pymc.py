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

# predictors vs. outcome 
d_pred = d_life.drop(columns = {'AMDELT'})
d_out = d_life['AMDELT']

# takes forever 
model = pm.Model(coords={"predictors": d_pred.columns.values})
with model: 
    alpha = pm.Normal("alpha", mu=0, sigma=5)
    beta = pm.Normal("beta", mu=0, sigma=5, dims="predictors")
    p = pm.Deterministic("p", pm.math.invlogit(alpha + at.dot(d_pred.values, beta)))
    outcome = pm.Bernoulli("outcome", p, observed=d_out.values)
with model: 
    idata = pm.sample(return_inferencedata=True)

az.summary(idata, var_names=['alpha', 'beta'], round_to=2)
idata.to_netcdf("../data/replication/full_model.nc")