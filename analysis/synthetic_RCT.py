import pandas as pd 
import numpy as np 

# loads 
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype=int)
configuration_probabilities = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')

# setup 
intervention_var = 11
outcome_var = 12

# pick something to vary 
intervention_on_idx = np.where(configurations[:, intervention_var] == 1)[0]
intervention_off_idx = np.where(configurations[:, intervention_var] == -1)[0]

# just start looking at the intervention off 
intervention_off_idx_sample = np.random.choice(intervention_off_idx, 1000, replace=False) # replace, probability?
# np.random.shuffle(A) # they are already random 

# Split A into two equal-sized arrays B and C
control_group_idx = intervention_off_idx_sample[:500]
experiment_group_idx = intervention_off_idx_sample[500:]

# Get the actual configurations 
control_group_configurations = configurations[control_group_idx]
experiment_group_configurations = configurations[experiment_group_idx]

# apply treatment 
experiment_group_configurations[:, intervention_var] = experiment_group_configurations[:, intervention_var]*-1

# flip the outcome variable 
control_group_flipped = np.copy(control_group_configurations)
control_group_flipped[:, outcome_var] = control_group_flipped[:, outcome_var]*-1
experiment_group_flipped = np.copy(experiment_group_configurations)
experiment_group_flipped[:, outcome_var] = experiment_group_flipped[:, outcome_var]*-1

# now get idx of control group flipped
# NB: this is a bottleneck
control_group_flipped_idx = np.array([np.where((configurations == i).all(1))[0][0] for i in control_group_flipped])
experiment_group_flipped_idx = np.array([np.where((configurations == i).all(1))[0][0] for i in experiment_group_flipped])

# now we can get probabilities & sample 
p_control_group = configuration_probabilities[control_group_idx]
p_control_group_flipped = configuration_probabilities[control_group_flipped_idx]
p_experiment_group = configuration_probabilities[experiment_group_idx]
p_experiment_group_flipped = configuration_probabilities[experiment_group_flipped_idx]

p_control_group = np.stack((p_control_group, p_control_group_flipped), axis = 1)
p_experiment_group = np.stack((p_experiment_group, p_experiment_group_flipped), axis = 1)

p_control_group_rowsum = p_control_group.sum(axis=1)
p_control_group_norm = p_control_group / p_control_group_rowsum[:, np.newaxis]

p_experiment_group_rowsum = p_experiment_group.sum(axis=1)
p_experiment_group_norm = p_experiment_group / p_experiment_group_rowsum[:, np.newaxis]

# now we need to get the possible outcomes
control_group_flipped_out = control_group_flipped[:, outcome_var]
control_group_out = control_group_configurations[:, outcome_var]
experiment_group_flipped_out = experiment_group_flipped[:, outcome_var]
experiment_group_out = experiment_group_configurations[:, outcome_var]

control_group_out = np.stack((control_group_out, control_group_flipped_out), axis = 1)  
experiment_group_out = np.stack((experiment_group_out, experiment_group_flipped_out), axis = 1)

# get probabilities for both all elements 
control_outcome = np.array([np.random.choice(row, p=probs) for row, probs in zip(control_group_out, p_control_group_norm)])
experiment_outcome = np.array([np.random.choice(row, p=probs) for row, probs in zip(experiment_group_out, p_experiment_group_norm)])

# now we gather one last time before modeling 
control_outcome
experiment_outcome

outcome = np.concatenate((control_outcome, experiment_outcome)) # also 0/1
outcome[outcome < 0] = 0  
intervention = np.concatenate((np.zeros(500), np.ones(500))) # or this -1/1

np.mean(control_outcome)
np.mean(experiment_outcome)


# now we are ready to model this 
import pymc as pm 
import arviz as az 

model = pm.Model()
with model: 
    alpha = pm.Normal("alpha", mu=0, sigma=5) # should mu=0 be baseline expectation?
    beta = pm.Normal("beta", mu=0, sigma=5)
    p = pm.Deterministic("p", pm.math.invlogit(alpha + intervention*beta))
    result = pm.Bernoulli("outcome", p, observed=outcome)
    
with model: 
    idata = pm.sample(
        draws = 1000, # default
        tune = 1000, # default
        target_accept = .99,
        max_treedepth = 20,
        random_seed = 412)

az.summary(idata, var_names = ['alpha', 'beta'], round_to=2) 

# does this give us anything besides what we already have?
# i.e. should this just re-capture the Jij?
# correlation here should also give us "causation"... 
# also what is the interpretation here? 
np.exp(1.04) 
# 2.8 times more likely for 13 to be on if 12 is on (correlation)
# 2.8 times more likely that we will move to 12 if 13 is on (causation)