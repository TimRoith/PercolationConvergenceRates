import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import graphlearning as gl
from matplotlib.pyplot import cm

#%%
def compute_optimal_path(predecessors, index):
    path = []
    while index != -9999:
        path.append(index)
        index = predecessors[0,index]
    return path    
 
#%% trials and distances
trials = 3
max_dists = 20
 
#%% dimensional constants
d = 2

#%%Simulation window parameters
s_max = 300

x1_min = -s_max/4
x1_max = 1.25 * s_max
x1_delta = x1_max - x1_min

x_min = -s_max**(1/d)
x_max = s_max**(1/d)
x_delta = x_max - x_min

area = x_delta**(d-1) * x1_delta
 
#Point process parameters
lamda = 1 #intensity (ie mean density) of the Poisson process

#%%
dists = np.linspace(2.1, s_max, max_dists)

#%% Plotting
def scale(s,d):
    # k = d + 2
    # return Cd*(k*np.log(2*Cdprime*Cd*s))**(1/d)
    return np.log(s)**(1/d)

#%%
ratios = [[] for T in range(trials)]

for T in range(trials):
    color = iter(cm.spring(np.linspace(0, 1, len(dists))))

    print(10*'<>')
    print('Trial {}'.format(T+1))
    print(10*'<>')
    np.random.seed(T)
    
    #construct Poisson cloud
    num_pts = scipy.stats.poisson(lamda * area).rvs()#Poisson number of points
    points = np.zeros((num_pts+2, d))

    for i in range(d):
        if i == 0:
            points[2:, i] = x1_delta * scipy.stats.uniform.rvs(0,1,((num_pts,)))+x1_min
        else:
            points[2:, i] = x_delta * scipy.stats.uniform.rvs(0,1,((num_pts,)))+x_min
    
    #plot Poisson cloud
    if d == 1:
        plt.figure()
        plt.scatter(points,0*points)
    if d == 2:
        plt.figure()
        plt.scatter(points[:,0],points[:,1])
    # if d == 3:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     # ax.scatter(points[:,0],points[:,1],points[:,2])
            
    
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
            path = compute_optimal_path(predecessors, 1)
            c = next(color)
            if d == 1:
                plt.scatter(points[idx[path]],0*points[idx[path]],color=c)
            if d == 2:
                plt.scatter(points[idx[path],0],points[idx[path],1],color=c)                
            # if d == 3:
            #     plt.scatter(points[idx[path],0],points[idx[path],1],points[idx[path],2],color=c)
                    
        
        print('Distance {}, ratio {}'.format(s,g_dist/s))
        ratios[T].append(g_dist/s)
        
        
#%% plotting
plt.figure()
for T in range(trials):
    plt.plot(ratios[T])

