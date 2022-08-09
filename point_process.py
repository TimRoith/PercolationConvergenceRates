import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import graphlearning as gl
 
#Simulation window parameters
x_min=0;
x_max=100;
d = 3

x_delta = x_max - x_min

area = x_delta**d
 
#Point process parameters
lamda = 1 #intensity (ie mean density) of the Poisson process
 
#%% Simulate Poisson point process
#y coordinates of Poisson points

#%%
dists = np.linspace(2.1, x_max, 50)


#%% Plotting

def scale(s,d):
    return np.log(s)**(1/d)

#%%

trials = 20
ratios = [[] for T in range(trials)]

for T in range(trials):
    np.random.seed(T)
    
    num_pts = scipy.stats.poisson(lamda * area).rvs()#Poisson number of points
    points = np.zeros((num_pts+2, d))
    
    for i in range(d):
        points[2:, i] = x_delta * scipy.stats.uniform.rvs(0,1,((num_pts,)))+x_min
    
    for s in dists:
        # update target point
        points[1,0] = s
        
        # get points in frame
        idx, = np.where(np.prod(np.abs(points) < (s+1), axis=1))
        loc_points = points[idx,:]
        
        h = scale(s, d)
        # get weight matrix
        W = gl.weightmatrix.epsilon_ball(loc_points, h, kernel='distance')
        G = gl.graph(W)
        
        if np.sum(W[0,:]) == 0:
            g_dist = np.inf
        else:
            u = scipy.sparse.csgraph.dijkstra(W, directed=False, indices=[0])
            g_dist = u[0,1]
        
        print(g_dist/s)
        ratios[T].append(g_dist/s)
        
        
#%% plotting
for T in range(trials):
    plt.plot(ratios[T])
