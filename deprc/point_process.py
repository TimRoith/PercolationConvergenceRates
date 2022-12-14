import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import graphlearning as gl
from matplotlib.pyplot import cm

#%% custom imports
from utils import compute_optimal_path, scale, plot_handler
 
 
sc_plot = False
    
#%% trials and distances
trials = 20
max_dists = 20
 
#%% dimensional constants
d = 2

#%%Simulation window parameters
s_max = 200

x1_min = -s_max/4
x1_max = 1.25 * s_max
x1_delta = x1_max - x1_min

x_min = -s_max**(1/d)
x_max = s_max**(1/d)
x_delta = x_max - x_min

area = x_delta**(d-1) * x1_delta
 
#Point process parameters
lamda = 1 #intensity (ie mean density) of the Poisson process

#%% array for distances, points and scaling
dists = np.linspace(2.1, s_max, max_dists)
points = np.zeros((2, d))


def scale(s,d):
    return np.log(s)**(1/d)

#%% Plotting
class plot_handler:
    def __init__(self, ax, points):
        self.d = points.shape[-1]
        self.points=points
        
        if self.d==1:
            ax[0,0].scatter(points, 0*points, c='skyblue', s=1)
            self.path_vis = ax[0,0].scatter(points[:2], 0*points[:2], c='tab:pink',s=10)
            self.target_vis = ax[0,0].scatter(points[:2], 0*points[:2], c='navy', s=25, marker='o')

        elif self.d==2:
            ax[0,0].scatter(points[:,0], points[:,1], c='skyblue', s=1)
            path_plot, = ax[0,0].plot(points[:2,0], points[:2,1], c='tab:pink',linewidth=2)
            self.path_vis = path_plot
            self.target_vis = ax[0,0].scatter(points[:2,0], points[:2,1], c='navy', s=25, marker='o')
        
        elif self.d==3:
            ax[0,0].scatter(points[:,0], points[:,1], zs=points[:,2], c='skyblue', s=1,alpha=0.1)
            path_plot, = ax[0,0].plot(points[:2,0], points[:2,1], zs=points[:2,2], c='tab:pink',linewidth=2)
            self.path_vis = path_plot
            self.target_vis = ax[0,0].scatter(points[:2,0], points[:2,1], points[:2,2], c='navy', s=25, marker='o')
        
        else:
            raise ValueError('Unsupported dimension!')
            
    def update(self, idx):
        if self.d==1:
            self.target_vis.set_offsets(np.hstack([self.points[:2], np.zeros((2,1))]))
            if len(idx)>1:
                self.path_vis.set_offsets(np.hstack([self.points[idx],np.zeros((len(idx),1))]))
        elif self.d==2:
            self.target_vis.set_offsets(self.points[:2,:])
            self.path_vis.set_data(self.points[idx,0],self.points[idx,1])
            
        elif self.d==3:
            pass
            self.target_vis._offsets3d = (self.points[:2,0], self.points[:2,1], self.points[:2,2]) 
            self.path_vis.set_data_3d(self.points[idx,0], self.points[idx,1], self.points[idx,2])


#%% Plotting
sc_plot = True
                
#%% main loop
ratios = np.zeros((trials, max_dists))

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
    
    # set up plots
    if sc_plot:
        plt.close('all')
        proj = '3d' if d==3 else None
        fig, ax = plt.subplots(1,1,squeeze=False, subplot_kw={'projection': proj})
        
        ph = plot_handler(ax, points)
        plt.show()
        plt.pause(0.1)
            
    
    for i in range(max_dists):
        s = dists[i]
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
            dist_matrix, predecessors = scipy.sparse.csgraph.shortest_path(W, directed=False, indices=[0],return_predecessors=True)
            g_dist = dist_matrix[0,1]
            path = compute_optimal_path(predecessors, 1)
            c = next(color)
            
            # scatter plot
            if sc_plot:
                ph.update(idx[path])
                plt.draw()
                # plt.pause(0.5)   
        
        print('Distance {}, ratio {}'.format(s,g_dist/s))
        ratios[T,i] = g_dist/s
        
exp_ratios = np.mean(ratios, axis=0)
        
#%% plotting
plt.figure()
idx = [i for i in range(len(dists)) if exp_ratios[i]<np.inf]
exp_errors = np.abs(exp_ratios - exp_ratios[-1])
p = np.polyfit(np.log(dists[idx][:-1]),np.log(exp_errors[idx][:-1]),1)
rate = p[0]
plt.loglog(dists[idx],exp_errors[idx])
plt.axis('equal')
print(10*'<>')
print('Rate of expected value: {}'.format(rate))
print(10*'<>')

plt.figure()
for T in range(trials):
    ratios_trial = np.array(ratios[T])
    errors = np.abs(ratios_trial - ratios_trial[-1])
    idx = [i for i in range(len(dists)) if ratios_trial[i]<np.inf]
    p = np.polyfit(np.log(dists[idx][:-1]),np.log(errors[idx][:-1]),1)
    rate = p[0]
    plt.loglog(dists[idx],errors[idx])
    plt.axis('equal')
    print('Rate in trial {}: {}'.format(T,rate))
