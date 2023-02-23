import numpy as np 
import pymc as pm 
import pandas as pd 
import argparse 
import pytensor.tensor as at 

def string_of_numbers_to_list(number_string): 
    return [int(m) for n in number_string for m in n if m in '0123456789']

def alpha_to_h(beta, alpha):
    return (beta/2 + alpha)/2

def alpha_transform(alpha_summary, beta_summary): 
    hi_list = []
    n_alpha = len(alpha_summary)
    for alpha_idx in range(n_alpha): 
        alpha_val = alpha_summary.iloc[alpha_idx]['mean']
        beta_sum = beta_summary[beta_summary['param'].str.contains(f"{alpha_idx}")]['mean'].sum()
        hi_value = alpha_to_h(beta_sum, alpha_val)
        hi_list.append(hi_value)
    return hi_list 

n_questions = 11
n_samples = 500
true_beta = 1.0
 
with open(f'../data/logistic/{n_questions}._{n_samples}._{true_beta}._data.dat') as f: 
    lines = f.readlines()
samples = [line.strip().split()[0] for line in lines[2:]]
X = np.array([string_of_numbers_to_list(x) for x in samples]) # needs generalization

# load actual params 
paramsTrue = np.loadtxt(f'../data/logistic/{n_questions}._{n_samples}._{true_beta}._params.dat')
Jtrue = paramsTrue[:n_questions*(n_questions-1)//2]
htrue = paramsTrue[n_questions*(n_questions-1)//2:]

# load simon params 
paramsMPF = np.loadtxt(f'../data/logistic/{n_questions}._{n_samples}._{true_beta}._data.dat_params.dat')
JMPF = paramsMPF[:n_questions*(n_questions-1)//2]
hMPF = paramsMPF[n_questions*(n_questions-1)//2:]

# generate the Y ~ X1 + X2 + X3 + ... + Xn setups 
questions = [x for x in range(n_questions)]
predictor_idx_list = []
outcome_idx_list = []
for outcome in questions:
    predictors = [x for x in questions if x != outcome]
    predictor_idx_list.append(predictors)
    outcome_idx_list.append(outcome)

# fit all combinations of models  (number 2)
idata_list = []
combination_list = []
for predictor_idx, outcome_idx in zip(predictor_idx_list, outcome_idx_list):
    print(outcome_idx)
    model = pm.Model()
    with model: 
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=len(predictor_idx))
        p = pm.Deterministic("p", pm.math.invlogit(alpha + at.dot(X[:, predictor_idx], beta)))
        outcome = pm.Bernoulli("outcome", p, observed=X[:, outcome_idx])
    with model: 
        idata = pm.sample(
            draws = 1000, # default
            tune = 1000, # default
            target_accept = .99,
            max_treedepth = 20,
            random_seed = 412
        )
    idata_list.append(idata)
    combination_list.append([(predictor_idx), (outcome_idx)])

# wrangle data 
# we should look into coords and dims 
n_draws = 1000
n_chains = 4
b_list = []
a_list = []
for num, ele in enumerate(combination_list): 
    # setup 
    idata = idata_list[num]
    predictors = ele[0]
    outcome = ele[1]
    
    # take out idata
    a_posterior = idata.posterior.alpha.values.reshape(n_draws*n_chains)
    b_posterior = idata.posterior.beta.values.reshape(n_draws*n_chains, n_questions-1)
    
    # gather beta values 
    for num, pred in enumerate(predictors):
        small, large = sorted([pred, outcome])
        b_posterior_df = pd.DataFrame({'value': b_posterior[:, num]})
        b_posterior_df['param'] = f'beta_{small}_{large}'
        b_list.append(b_posterior_df)    
        
    # gather alpha values 
    a_posterior_df = pd.DataFrame(a_posterior, columns = ["value"])
    a_posterior_df['param'] = f"alpha_{outcome}"
    a_list.append(a_posterior_df)
idata.sample_stats
# get the dataframes 
a_samples = pd.concat(a_list)
b_samples = pd.concat(b_list)

# aggregate 
b_summary = b_samples.groupby('param').value.agg(['mean', 'std', 'median', 'min', 'max']).reset_index()
a_summary = a_samples.groupby('param').value.agg(['mean', 'std', 'median', 'min', 'max']).reset_index()

a_summary['transform'] = alpha_transform(a_summary, b_summary)
b_summary['transform'] = b_summary['mean']/4

# put them together
full_summary = pd.concat([a_summary, b_summary])
full_summary['MPF'] = np.concatenate([hMPF, JMPF])
full_summary['Actual'] = np.concatenate([htrue, Jtrue])

# save this information 
full_summary.to_csv(f'../data/logistic/summary_sigma10_{n_questions}_{n_samples}_{true_beta}.csv', index=False)
