import numpy as np 
import matplotlib.pyplot as plt

# plotting function
def plot_params(params_true,
                params_inf,
                title,
                constant):
    min_lim = np.min(np.concatenate((params_true, params_inf))) - constant
    max_lim = np.max(np.concatenate((params_true, params_inf))) + constant
    plt.scatter(params_true, params_inf)
    plt.plot([min_lim, max_lim], [min_lim, max_lim], 'k--')
    plt.xlabel('true')
    plt.ylabel('inferred')
    plt.suptitle(title)

def plot_h_hidden(params_true,
                  params_inf,
                  n_hidden,
                  title,
                  constant):
    min_lim = np.min(np.concatenate((params_true, params_inf))) - constant
    max_lim = np.max(np.concatenate((params_true, params_inf))) + constant
    params_true_hidden = params_true[:n_hidden]
    params_inf_hidden = params_inf[:n_hidden]
    params_true_visible = params_true[n_hidden:]
    params_inf_visible = params_inf[n_hidden:]
    plt.scatter(params_true_visible, params_inf_visible, color='tab:blue', label='visible')
    plt.scatter(params_true_hidden, params_inf_hidden, color='tab:orange', label='hidden')
    plt.plot([min_lim, max_lim], [min_lim, max_lim], 'k--')
    plt.xlabel('true')
    plt.ylabel('inferred')
    plt.suptitle(title)