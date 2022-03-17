import load_lib
import os
import numpy as np
import sys
from mayavi import mlab

import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import pandas as pd
 
def get_index(vec, sample):
    min_distance = np.Inf
    index = None
    for i in range(len(vec)):
        if abs(vec[i] - sample) < min_distance:
            min_distance = abs(vec[i] - sample)
            index = i #[i for i in range(len(dtheta)) if abs(dtheta[i] - aoa_theta) <=  1e-03]
    return index

def norm(vec):
    return np.sqrt(np.sum(vec ** 2))

def get_u(paths, dphi, dtheta):
    u_rx = np.zeros((len(dphi), len(dtheta)))
    u_tx = np.zeros((len(dphi), len(dtheta)))

    pwr = paths[:,4]
    pwr_norm = norm (pwr)
    #print (f'norm: {norm}')
    print (f'len: {len(pwr)}')
    print (f'norm of pwr: {pwr_norm}')

    for p in paths:
        rcvd_pwr =  p[4]/pwr_norm #10 * np.log(p[4])

        # at RX
        aoa_phi = p[1]
        #aoa_phi = np.rad2deg(p[1])
        index_dphi_rx = get_index(dphi, aoa_phi)
        aoa_theta = p[0]
        #aoa_theta = np.rad2deg(p[0])
        index_dtheta_rx = get_index(dtheta, aoa_theta)
        u_rx[index_dphi_rx, index_dtheta_rx] += rcvd_pwr

        # at TX
        aod_phi = p[3]
        #aod_phi = np.rad2deg(p[3])
        index_dphi_tx = get_index(dphi, aod_phi)
        aod_theta = p[2]
        #aod_theta = np.rad2deg(p[2])
        index_dtheta_tx = get_index(dtheta, aod_theta)
        u_tx[index_dphi_tx, index_dtheta_tx] += rcvd_pwr

    return u_rx, u_tx
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


if __name__ == '__main__':

    npy_pathfile1 = sys.argv[1]
    npy_pathfile2 = sys.argv[2]

    paths1 = np.load(npy_pathfile1)
    paths1 = paths1.T
    paths2 = np.load(npy_pathfile2)
    paths2 = paths2.T

    #paths1 = np.load(npy_pathfile1) allow_pickle=True)
    #paths2 = np.load(npy_pathfile2) allow_pickle=True)
    #paths1 = paths1[23]
    print (np.shape(paths1))
    #paths2 = paths2[45]
    print (np.shape(paths2))

    gridsize = (2, 2)
    fig = plt.figure(figsize=(7,7))
    
    ax1 = plt.subplot2grid(gridsize, (0, 0))
    ax1.set_title('RX4')

    ax2 = plt.subplot2grid(gridsize, (0, 1))
    ax2.set_title('TX(RX4)')

    ax3 = plt.subplot2grid(gridsize, (1, 0))
    ax3.set_title('RX5')

    ax4 = plt.subplot2grid(gridsize, (1, 1))
    ax4.set_title('TX(RX5)')

    dphi = np.linspace(-180, 180, 81) # index
    dtheta = np.linspace(180, 0, 46) # columns

    u_rx1, u_tx1 = get_u(paths1, dphi, dtheta)
    u_rx2, u_tx2 = get_u(paths2, dphi, dtheta)

    x_min = dphi[0]
    x_max = dphi[-1]

    y_min = dtheta[0]
    y_max = dtheta[-1]


    z_min = 0
    z_max = np.max(u_rx1.flatten())

    mlab.barchart(u_rx1)

    mlab.xlabel('phi')

    mlab.ylabel('theta')
    mlab.zlabel('gain')
    #mlab.barchart(u_rx2)

    #mlab.imshow(u_rx1)
    mlab.axes(ranges=[x_min, x_max, y_min, y_max, z_min, z_max])
    mlab.colorbar()
    mlab.show()

#    #print (u_rx1)
#    #print (np.shape(u_rx1)) 
#
#    #print (u_tx1)
#    #print (np.shape(u_tx1)) 
#    
#    df1_rx = pd.DataFrame(u_rx1.T, index=dtheta, columns=dphi)
#    df1_tx = pd.DataFrame(u_tx1.T, index=dtheta, columns=dphi)
#    df2_rx = pd.DataFrame(u_rx2.T, index=dtheta, columns=dphi)
#    df2_tx = pd.DataFrame(u_tx2.T, index=dtheta, columns=dphi)
#    #print (f'df: {df}')
#    #df1 = pd.DataFrame(u_rx1, index=dtheta, columns=dphi)
#    #df2 = pd.DataFrame(u_tx1, index=dtheta, columns=dphi)
#
#    #df3 = pd.DataFrame(u_rx2, index=dtheta, columns=dphi)
#    #df4 = pd.DataFrame(u_tx2, index=dtheta, columns=dphi)
#    #print ('u:\n')
#    #print (u)
#    #print ('df:\n')
#    #print (df1)
#    #sns.heatmap(df, ax=ax1)
#    sns.heatmap(df1_rx, ax=ax1, cmap='PiYG')
#    sns.heatmap(df1_tx, ax=ax2, cmap='PiYG')
#    sns.heatmap(df2_rx, ax=ax3, cmap='PiYG')
#    sns.heatmap(df2_tx, ax=ax4, cmap='PiYG')
#    #sns.heatmap(df2, ax=ax2)
#    #sns.heatmap(df3, ax=ax3)
#    #sns.heatmap(df4, ax=ax4)
#    #df1.plot.scatter(subplots=True, ax=ax1, sharex=True, sharey=True, title=['RX4'], xlabel='phi', ylabel='theta', grid=True, x=['aoa phi'], y=['aoa theta'])
#
#    ##df2 = pd.DataFrame(paths2.T, columns=['aoa theta', 'aoa phi', 'aod theta', 'aod phi', 'received power'])
#    ##sns.heatmap(df2, sharex=True, ax=ax2)
#    #df2.plot.scatter(subplots=True, ax=ax2, sharex=True, sharey=True, title=['RX5'], xlabel='phi', ylabel='theta', grid=True, x=['aoa phi'], y=['aoa theta'])
# 
#    #print (df)
#    #plt.legend()
#    plt.show()
