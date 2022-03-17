import load_lib
import os
import numpy as np
import sys
import json

import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import pandas as pd
from mayavi import mlab #import barchart, plot3d
from lib.utils.plot import plot_devices
 
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


if __name__ == '__main__':

    
    path_info_filename = 'all_paths_data_from_007.csv' 
    df = pd.read_csv(path_info_filename)
    #with open(device_pos_info_filename) as device_pos_info:
    #    data = device_pos_info.read()
    #    d = json.loads(data)
    gridsize = (2, 2)
    fig = plt.figure(figsize=(700,800))
        
    #ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
    #ax1.set_title("my title")

    #only_los_paths = df[(df['los'] == 1) & (df['episode'] == 0) & (df['scene'] == 0) & (df['channel'] == 1)]
    only_los_paths = df[(df['episode'] == 0) & (df['scene'] == 0) & (df['channel'] == 1)]

    departure_theta = only_los_paths['departure_theta']
    departure_theta = departure_theta.to_numpy()

    departure_phi = only_los_paths['departure_phi']
    departure_phi = departure_phi.to_numpy()

    arrival_theta = only_los_paths['arrival_theta']
    arrival_theta = arrival_theta.to_numpy()

    arrival_phi = only_los_paths['arrival_phi']
    arrival_phi = arrival_phi.to_numpy()

    received_power = only_los_paths['received_power'].to_numpy() 
    #received_power = received_power/norm(received_power)
    #received_power = len(only_los_paths) * received_power

    ax1 = plt.subplot2grid(gridsize, (0, 0), projection='polar')
    plt.title('departure_theta')
    plt.polar(departure_theta, received_power, 'g.')

    ax2 = plt.subplot2grid(gridsize, (0, 1), projection='polar')
    plt.title('departure_phi')
    plt.polar(departure_phi, received_power, 'g.')
    #for i in range(len(only_los_paths)): # in departure_theta:
    #    plt.arrow(departure_phi[i], 0, 0, received_power[i], alpha = 0.5, width = 0.015, edgecolor = 'black', facecolor = 'green', lw = 2, zorder = 5)

    ax3 = plt.subplot2grid(gridsize, (1, 0), projection='polar')
    plt.title('arrival_theta')
    plt.polar(arrival_theta, received_power, 'r.')
    #for i in range(len(only_los_paths)): # in departure_theta:
    #    plt.arrow(arrival_theta[i], 0, 0, received_power[i], alpha = 0.5, width = 0.015, edgecolor = 'black', facecolor = 'green', lw = 2, zorder = 5)
    #plt.arrow(arrival_theta, 0, 0, 0.85, alpha = 0.5, width = 0.015, edgecolor = 'black', facecolor = 'green', lw = 2, zorder = 5)
    #plt.polar(arrival_theta, received_power, "bs")

    ax4 = plt.subplot2grid(gridsize, (1, 1), projection='polar')
    plt.title('arrival_phi')
    plt.polar(arrival_phi, received_power, 'r.')
    #for i in range(len(only_los_paths)): # in departure_theta:
    #    plt.arrow(arrival_phi[i], 0, 0, received_power[i], alpha = 0.5, width = 0.015, edgecolor = 'black', facecolor = 'green', lw = 2, zorder = 5)
    #plt.arrow(arrival_phi, 0, 0, 0.85, alpha = 0.5, width = 0.015, edgecolor = 'black', facecolor = 'green', lw = 2, zorder = 5)
    #plt.polar(arrival_phi, received_power, "bs")

    #plt.savefig('aod_aod_paths.png')
    plt.show()

    xyz = only_los_paths[['x', 'y', 'z']]
    xyz = xyz.to_numpy()
    x_pos = xyz[0,0]
    y_pos = xyz[0,1]
    z_pos = xyz[0,2]

    transceivers = []

    device0 = 'tx'
    dev0_label = '1'
    dev0_posx = 746 
    dev0_posy = 560
    dev0_posz = 4
    transceiver1 = {'type': device0, 'label': dev0_label, 'posx': dev0_posx, 'posy': dev0_posy, 'posz': dev0_posz} 
    transceivers.append(transceiver1)

    device1 = 'rx'
    dev1_label = '1'
    dev1_posx = x_pos
    dev1_posy = y_pos
    dev1_posz = z_pos
    transceiver1 = {'type': device1, 'label': dev1_label, 'posx': dev1_posx, 'posy': dev1_posy, 'posz': dev1_posz} 
    transceivers.append(transceiver1)
#
#
    x = []
    y = []
    z = []
    label = []
    for transceiver in transceivers:
        print (transceiver)
        x.append(transceiver['posx'])
        y.append(transceiver['posy'])
        z.append(transceiver['posz'])
        label.append(transceiver['type']+transceiver['label'])
        
#    #plot_devices(transceivers)
#
    x_min = 0
    x_max = 900
    
    y_min = 0
    y_max = 900

    z_min = 0
    z_max = 900

    #s = [1 for i in range(len(transceivers))]
    #mlab.plot3d([x[0],x[1]], [y[0],y[1]], [z[0],z[1]], tube_radius=0.25)
    #mlab.points3d(x, y, z, colormap="copper", scale_factor=5)
    #mlab.axes(ranges=[x_min, x_max, y_min, y_max, z_min, z_max]) 
    #for i in range(len(label)):
    #    mlab.text3d(x[i]+5, y[i]+5, z[i]+5, label[i], scale=5)
    #mlab.view(azimuth=180, elevation=0)
    #mlab.view(azimuth=90, elevation=None)
    ##mlab.show()
