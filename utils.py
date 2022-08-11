import numpy as np
import scipy

#%% extract optimal path
def compute_optimal_path(predecessors, index):
    path = []
    while index != -9999:
        path.append(index)
        index = predecessors[0,index]
    return path

#%% scale function
def scale(s,d):
    # k = d + 2
    # return Cd*(k*np.log(2*Cdprime*Cd*s))**(1/d)
    return np.log(s)**(1/d)

#%% point process
class Poisson_process:
    def __init__(self, bounds, d = 1, lamda = 1.0, area = 1.0):
        self.d = d
        self.lamda = lamda
        self.area = area
        self.bounds = bounds
    
    def __call__(self):
        num_pts = scipy.stats.poisson(self.lamda * self.area).rvs() #Poisson number of points
        points = np.zeros((num_pts, self.d))

        for i in range(self.d):
            points[:, i] = np.random.uniform(low=self.bounds[i, 0],\
                                             high=self.bounds[i, 1],\
                                             size=(num_pts,))
        return points

#%% plot handling
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
