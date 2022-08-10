import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import graphlearning as gl
from matplotlib.pyplot import cm
#%% custom imports
from utils import compute_optimal_path, scale, plot_handler, Poisson_process

#%% parameters
s_max = 500
max_dists = 20
dists = np.linspace(2.1, s_max, max_dists)
d = 2

bounds = np.zeros((d, 2))
bounds[0,0] = -s_max/4
bounds[0,1] = 1.25 * s_max

bounds[1:,0] = -s_max**(1/d)
bounds[1:,1] = s_max**(1/d)

delta = bounds[:,1] - bounds[:,0]
area = np.prod(delta)
lamda = 1 #intensity (ie mean density) of the Poisson process
random_seed = 42

#%% 
trials = 20
ratios = [[] for i in range(trials)]
PP = Poisson_process(bounds, lamda = lamda, area = area, d=d)


#%% main loop
for T in range(trials):
    print(10*'<>')
    print('Trial {}'.format(T+1))
    print(10*'<>')
    
    np.random.seed(T)
    points = np.vstack([np.zeros((2, d)), PP()])
    
    for s in dists:
        # update target point
        points[1,0] = s
        
        # get points in frame
        idx, = np.where(np.prod(np.abs(points) < 1.25*s, axis=1))
        loc_points = points[idx,:]
        
        h = scale(s, d)
        # get weight matrix
        W = gl.weightmatrix.epsilon_ball(loc_points, h, kernel='distance')
        G = gl.graph(W)
        
        if np.sum(W[0,:]) == 0:
            g_dist = np.inf
        else:
            # dist_matrix = scipy.sparse.csgraph.dijkstra(W, directed=False, indices=[0])
            dist_matrix, predecessors = scipy.sparse.csgraph.shortest_path(W, directed=False, indices=[0],return_predecessors=True)
            g_dist = dist_matrix[0,1] 
        
        print('Distance {}, ratio {}'.format(s,g_dist/s))
        ratios[T].append(g_dist/s)