import sys
import load_lib
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import pandas as pd
import lib.mimo.arrayconfig as arraycfg
from utils import norm
from numpy.linalg import svd
#from mayavi import mlab

#x = np.arange(-5, 5, 0.1)
#y = np.arange(-5, 5, 0.1)

#xx, yy = np.meshgrid(x, y, sparse=True)
#z = np.sin(xx**2 + yy**2) #  / (xx**2 + yy**2)
#print (f'shape: {np.shape(z)}')

#uniform_data = np.random.rand(10, 12)

#ax = sns.heatmap(z)
#ax.get_figure().savefig('heatmap.png')



if __name__ == "__main__":

    #samples_npy_file = sys.argv[1]
    #samples = np.load(samples_npy_file)

    precision = 300
    dtheta = np.linspace(-90, 90, precision)
    dphi = np.linspace(-180, 180, precision)

    ##dphi = np.arange(-180, 181)
    ##dtheta = np.arange(-180, 181)

    phi, theta = np.meshgrid(dphi, dtheta)

    u = np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(phi)) # u
    #v = np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi)) # v

    
    # View it.
    print (f'u.shape: {np.shape(u)}')
    df = pd.DataFrame(u, index=dtheta, columns=dphi)
    ax = sns.heatmap(df, cmap='coolwarm')
    #ax.get_figure().savefig('heatmap.png')
    plt.ylabel("theta")
    plt.xlabel("phi")
    plt.show()
import os
import numpy as np

#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/database')
#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/mimo')
#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/vq')
import load_lib
from database.preprocessing import check_hdf5files
from database.getsamples import get_paths_by_receiver

import database.dataconfig as datacfg
import mimo.arrayconfig as arraycfg

import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import pandas as pd
 
prefix_episodefiles = datacfg.prefix_episodefiles
episode_files = os.listdir(prefix_episodefiles)
pathfiles = check_hdf5files(prefix_episodefiles, episode_files)
print (len(pathfiles))
path_set = get_paths_by_receiver(None, pathfiles)
    
#    rx = 5
for rx_id, paths in path_set.items():
    print (f'Receiver: RX{rx_id}')
    #fig, axes = plt.subplots(subplot_kw=dict(polar=True))
    ##paths = np.random.choice(paths, 1000)
    #for p in paths:
    #    pwr_rcv = pwr = p['received_power']
    #    aoa_theta = np.rad2deg(p['arrival_theta'])
    #    aoa_phi = np.rad2deg(p['arrival_phi'])
    #    axes.plot(aoa_theta, pwr_rcv, 'ro')
    #plt.show()
    #paths = path_set[rx]
    #paths = v
    #print (f'len(paths: {len(paths)})')
    
    
    precision = 10
    dtheta = np.linspace(0, 180, precision)
    dphi = np.linspace(-180, 180, precision)
    
    #phi, theta = np.meshgrid(dphi, dtheta)
    
    u = np.zeros((len(dtheta), len(dphi)))
    
    #paths = np.random.choice(paths, 2)
    aoa_theta = []
    aoa_phi = []
    pwr = []
    for p in paths:
        #aoa_theta = np.random.choice(dtheta) #np.rad2deg(p['arrival_theta'])
        aoa_theta.append(np.rad2deg(p['arrival_theta']))
        #aoa_phi = np.random.choice(dphi) #np.rad2deg(p['arrival_phi'])
        aoa_phi.append(np.rad2deg(p['arrival_phi']))
        pwr.append(p['received_power'])
        #print (f'aoa_theta, aoa_phi, pwr: {aoa_theta}, {aoa_phi}, {pwr}')

        #min_distance_theta = np.Inf
        #theta_index = None
        #for i in range(len(dtheta)):
        #    if abs(dtheta[i] - aoa_theta) < min_distance_theta:
        #        min_distance_theta = abs(dtheta[i] - aoa_theta)
        #        theta_index = i #[i for i in range(len(dtheta)) if abs(dtheta[i] - aoa_theta) <=  1e-03]
        ##print (f'theta_index: {theta_index}')

        #min_distance_phi = np.Inf
        #phi_index = None
        #for j in range(len(dphi)):
        #    if abs(dphi[j] - aoa_phi) < min_distance_phi:
        #        min_distance_phi = abs(dphi[j] - aoa_phi)
        #        phi_index = j #[i for i in range(len(dtheta)) if abs(dtheta[i] - aoa_theta) <=  1e-03]
        ##print (f'phi_index: {phi_index}')
        #u[theta_index, phi_index] += pwr
    #
    #u = np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(phi)) # u
        #v = np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi)) # v
    
        
    # View it.
    ##print (f'u.shape: {np.shape(u)}')
    ##df = pd.DataFrame(u, index=dtheta, columns=dphi)
    ##ax = sns.heatmap(df, cmap='coolwarm')
    #ax.get_figure().savefig('heatmap.png')
    ##plt.ylabel("theta")
    ##plt.xlabel("phi")
    ##plt.show()
    plt.scatter(aoa_phi, aoa_theta)
    plt.show()
