import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import graphlearning as gl
from matplotlib.pyplot import cm
import time
import pickle
#%% custom imports
from utils import compute_optimal_path, scale, plot_handler, Poisson_process

#%% parameters
s_max = 1000

num_dists = 20
dists = np.linspace(2.1, s_max/2, num_dists)
d = 2

bounds = np.zeros((d, 2))
bounds[0,0] = -s_max/4
bounds[0,1] = 1.25 * s_max

bounds[1:,0] = -s_max**(1/d)
bounds[1:,1] = s_max**(1/d)

delta = bounds[:,1] - bounds[:,0]
area = np.prod(delta)
lamda = 1 #intensity (ie mean density) of the Poisson process
random_seed = 142

#%% 
trials = 10
dists_1 = np.zeros((trials, len(dists)))
dists_2 = np.zeros((trials, len(dists)))
PP = Poisson_process(bounds, lamda = lamda, area = area, d=d)


#%% main loop
for T in range(trials):
    print(10*'<>')
    print('Trial {}'.format(T+1))
    print(10*'<>')
    
    np.random.seed(T)
    points = np.vstack([np.zeros((3, d)), PP()])
    
    for i,s in enumerate(dists):
        # update target points
        points[1,0] = s
        points[2,0] = 2*s
        
        # get points in frame
        idx, = np.where(np.prod(np.abs(points) < 2.25*s, axis=1))
        loc_points = points[idx,:]
        
        #h = scale(s, d)
        h = .7*np.log(s)**(1/d)
        #h = max(.01*s, 1.5)
        
        
        # get weight matrix
        W = gl.weightmatrix.epsilon_ball(loc_points, h, kernel='distance')
        G = gl.graph(W)
        
        if np.sum(W[0,:]) == 0:
            g_dist_1 = np.inf
            g_dist_2 = np.inf
        else:
            # dist_matrix = scipy.sparse.csgraph.dijkstra(W, directed=False, indices=[0])
            dist_matrix, predecessors = scipy.sparse.csgraph.shortest_path(W, directed=False, indices=[0],return_predecessors=True)
            g_dist_1 = dist_matrix[0,1]
            g_dist_2 = dist_matrix[0,2] 
        
        dists_1[T,i] = g_dist_1
        dists_2[T,i] = g_dist_2
        
#%% handle ratios
exp_dists_1 = np.mean(dists_1, axis=0)
exp_dists_2 = np.mean(dists_2, axis=0)

exp_ratios = exp_dists_1/exp_dists_2
exp_errors = np.abs(exp_ratios - 0.5)

idx = [i for i in range(len(dists)) if exp_ratios[i]<np.inf]
p = np.polyfit(np.log(dists[idx][:-1]),np.log(exp_errors[idx][:-1]),1)

#%% plotting
plt.close('all')
plt.loglog(dists, exp_errors)
plt.loglog(dists[idx][:-1],np.exp(p[1])*dists[idx][:-1]**p[0])

#%% save ratios to file
time_str = time.strftime("%Y%m%d-%H%M%S")
fname = "results/ratios-" + time_str + '.pkl'

with open(fname, 'wb') as f:
    # Put them in the file 
    pickle.dump([dists, dists_1, dists_2], f)

np.savetxt(fname, np.vstack([dists_1,dists_2]))



