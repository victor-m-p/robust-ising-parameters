import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# load configurations
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype = int)
configuration_probabilities = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')

# find the questions 
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')

# try official political support 
idx_pol_sup = np.where(configurations[:, 0] == 1)[0]
idx_not_pol_sup = np.where(configurations[:, 0] == -1)[0]
p_pol_sup = configuration_probabilities[idx_pol_sup]
p_not_pol_sup = configuration_probabilities[idx_not_pol_sup]

# first look overall 
p_pol_sup_norm = np.sort(p_pol_sup)[::-1] / np.max(p_pol_sup)
p_not_pol_sup_norm = np.sort(p_not_pol_sup)[::-1] / np.max(p_not_pol_sup)

n_states = len(p_pol_sup_norm)
plt.plot([np.log(i) for i in range(n_states)], 
         [np.log(i) for i in p_pol_sup_norm], color = 'tab:blue')
plt.plot([np.log(i) for i in range(n_states)],
         [np.log(i) for i in p_not_pol_sup_norm], color = 'tab:orange')

# stability instead of probability 
idx_stability = np.loadtxt('../data/analysis/idx_stability.txt', dtype = int)
np.all(np.diff(idx_stability) > 0) # sanity check

stability = np.loadtxt('../data/analysis/stability.txt')
stability_pol_sup = stability[idx_pol_sup]
stability_not_pol_sup = stability[idx_not_pol_sup]

# difference is so negligible 
np.mean(stability_pol_sup)
np.mean(stability_not_pol_sup)

# again, really not much there 
np.sum(stability_pol_sup)
np.sum(stability_not_pol_sup)

## loop over all of them 
lst = [] 
for num, ele in enumerate(question_reference['question']):
    idx_on = np.where(configurations[:, num] == 1)[0]
    idx_off = np.where(configurations[:, num] == -1)[0]
    p_on = np.mean(configuration_probabilities[idx_on])
    p_off = np.mean(configuration_probabilities[idx_off])
    psum_on = np.sum(configuration_probabilities[idx_on])
    psum_off = np.sum(configuration_probabilities[idx_off])
    stability_on = np.mean(stability[idx_on])
    stability_off = np.mean(stability[idx_off])
    stabilitysum_on = np.sum(stability[idx_on])
    stabilitysum_off = np.sum(stability[idx_off])
    lst.append([num, ele, p_on, p_off, psum_on, psum_off, stability_on, stability_off, stabilitysum_on, stabilitysum_off])
d = pd.DataFrame(lst, 
                 columns = ['num', 'question', 'p_on', 'p_off', 
                            'psum_on', 'psum_off',
                            'stability_on', 'stability_off',
                            'stabilitysum_on', 'stabilitysum_off'])

# for top configurations 
n = 1000
for num, ele in enumerate(question_reference['question']):
    idx_on = np.where(configurations[:, num] == 1)[0]
    idx_off = np.where(configurations[:, num] == -1)[0]
    stability_on = stability[idx_on]
    stability_off = stability[idx_off]
    plt.plot([np.log(i) for i in range(n)], 
            [i for i in np.sort(stability_on)[::-1][0:n]], color = 'tab:blue')
    plt.plot([np.log(i) for i in range(n)],
            [i for i in np.sort(stability_off)[::-1][0:n]], color = 'tab:orange')
    plt.title(ele)
    plt.savefig('../figures/stabilizers/' + str(num) + '.png')
    plt.clf()
