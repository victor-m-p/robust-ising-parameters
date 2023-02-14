import numpy as np
from fun import p_dist, bin_states, fast_logsumexp

# setup
n = 3
C = 100
np.random.seed(0)

# generate params and probabilities
h = np.random.normal(scale=0.5, size=n)
J = np.random.normal(scale=0.5, size=n*(n-1)//2)
hJ = np.concatenate((h, J))
probabilities = p_dist(h, J)

# visualize data 
allstates = bin_states(n, True)  # all 2^n possible binary states in {-1,1} basis
sample = allstates[np.random.choice(range(2**n), # doesn't have to be a range
                                    size=C, # how many samples
                                    replace=True, # a value can be selected multiple times
                                    p=probabilities)]  # random sample from p(s)

# calculate mean and correlation
means = np.mean(sample, axis=0)
correlations = np.corrcoef(sample, rowvar=False)
correlations
corr_list = []
for i, _ in enumerate(correlations): 
    for j, _ in enumerate(correlations): 
        if i > j: 
            corr_list.append(correlations[i,j])
correlations = np.array(corr_list)

# calculate for n=1
params = np.array([0, 0, 0, 0, 0, 0])

def calc_observables(params):
    """
    Give all parameters concatenated into one array from lowest to highest order.
    Returns all correlations.
    """
    Cout = np.zeros((6))
    H = params[0:3]
    J = params[3:6]
    energyTerms = np.array([    +H[0]+H[1]+H[2]+J[0]+J[1]+J[2], +H[0]+H[1]-H[2]+J[0]-J[1]-J[2], +H[0]-H[1]+H[2]-J[0]+J[1]-J[2], +H[0]-H[1]-H[2]-J[0]-J[1]+
    J[2], -H[0]+H[1]+H[2]-J[0]-J[1]+J[2], -H[0]+H[1]-H[2]-J[0]+J[1]-J[2], -H[0]-H[1]+H[2]+J[0]-J[1]-J[2], -H[0]-H[1]-H[2]+
            J[0]+J[1]+J[2],])
    logZ = fast_logsumexp(energyTerms)[0]
    num = fast_logsumexp(energyTerms, [ 1, 1, 1, 1,-1,-1,-1,-1])
    Cout[0] = np.exp( num[0] - logZ ) * num[1]
    num = fast_logsumexp(energyTerms, [ 1, 1,-1,-1, 1, 1,-1,-1])
    Cout[1] = np.exp( num[0] - logZ ) * num[1]
    num = fast_logsumexp(energyTerms, [ 1,-1, 1,-1, 1,-1, 1,-1])
    Cout[2] = np.exp( num[0] - logZ ) * num[1]
    num = fast_logsumexp(energyTerms, [ 1, 1,-1,-1,-1,-1, 1, 1])
    Cout[3] = np.exp( num[0] - logZ ) * num[1]
    num = fast_logsumexp(energyTerms, [ 1,-1, 1,-1,-1, 1,-1, 1])
    Cout[4] = np.exp( num[0] - logZ ) * num[1]
    num = fast_logsumexp(energyTerms, [ 1,-1,-1, 1, 1,-1,-1, 1])
    Cout[5] = np.exp( num[0] - logZ ) * num[1]
    Cout[np.isnan(Cout)] = 0.
    return(Cout)

tst = calc_observables(sample[0])
sample