import numpy as np
import matplotlib.pyplot as plt
import csv
import os


#%% get files and extract data
cur_path = os.path.dirname(os.path.realpath(__file__))
path = cur_path + "\\log_scales_large"
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

data = []

for fname in files:
    G = np.genfromtxt(path + '\\' +  fname, delimiter=',', skip_header=9)
    data.append(G[:,1:])

num_trials = data[0].shape[0]//2
dists = data[0][0,:]
#%%
plt.close('all')

scale_factors = [.5, 0.7, .9, 1.1]
colors=['r', 'y', 'b', 'g','peru', 'tan', 'coral',  ]

fig, ax = plt.subplots(1,2, figsize=(15, 5))


for i in range(len(data)):
    graph_dists = data[i][:num_trials, :]
    ratios = graph_dists/dists
    
    exp_ratios = np.mean(ratios, axis=0)
    std_ratios = np.std(ratios, axis=0)
    
    errors = np.abs(exp_ratios - exp_ratios[-1])
    m = len(errors)
    k = m-5
    
    #EOC = np.log(errors[1:m-1]/errors[2:])/np.log(errors[:m-2]/errors[1:m-1])
    # plotting
    color = colors[i]
    ax[0].plot(dists, exp_ratios, marker ='.', label=str(scale_factors[i]), color=color)
    ax[0].fill_between(dists, exp_ratios - std_ratios,\
                           exp_ratios + std_ratios,\
                           alpha=0.3, edgecolor=None, color=color)
        
    ax[1].plot(dists[:k], errors[:k], marker ='.', label=str(scale_factors[i]), color=color)
    ax[1].fill_between(dists[:k], (errors - std_ratios)[:k],\
                           (errors + std_ratios)[:k],\
                           alpha=0.3, edgecolor=None, color=color)
    
    #idx = [i for i in range(m-2) if 0<EOC[i]<np.inf]
    #ax[2].plot(EOC[idx], color = color,marker = '.')

ax[1].set_yscale('log')
ax[1].set_xscale('log')   
legend = ax[0].legend()
legend.set_title('Factors')
fig.savefig("vis/exp_converegnce.pdf",bbox_inches="tight")

#%% ratio convergence
m = data[0].shape[1]
fig2, ax2 = plt.subplots(1,2, figsize=(15, 5))

idx_1 = []
idx_2 = []
for i in range(m):
    for j in range(i+1,m):
        if np.isclose(dists[j]/dists[i], 2.0):
            idx_1.append(i)
            idx_2.append(j)
            break
    
for i in range(len(data)):
    graph_dists_1 = data[i][:num_trials, :]
    exp_dists_1 = np.mean(graph_dists_1, axis=0)
    
    graph_dists_2 = data[i][num_trials:, :]
    exp_dists_2 = np.mean(graph_dists_2, axis=0)
    
    idx, = np.where((exp_dists_1 + exp_dists_2) < np.inf)
    
    if len(idx)>0:
        ratios = exp_dists_1[idx]/exp_dists_2[idx]
        errors = np.abs(ratios - 0.5)
        color = colors[i]
        ax2[0].plot(dists[idx], ratios, marker ='.', label=str(scale_factors[i]), color=color)
        ax2[1].plot(dists[idx], np.abs(ratios-0.5), marker ='.', label=str(scale_factors[i]),  color=color)

ax2[1].set_yscale('log')
ax2[1].set_xscale('log')
legend = ax2[0].legend()
legend.set_title('Factors')

fig2.savefig("vis/ratio_converegnce.pdf",bbox_inches="tight")