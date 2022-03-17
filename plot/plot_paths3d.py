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

    device_pos_info_filename = 'device_pos_info.json' 
    with open(device_pos_info_filename) as device_pos_info:
        data = device_pos_info.read()
        d = json.loads(data)

    transceivers = []

    device0 = d['device0']
    dev0_label = d['dev0_label']
    dev0_posx = d['dev0_posx']
    dev0_posy = d['dev0_posy']
    dev0_posz = d['dev0_posz']
    transceiver1 = {'type': device0, 'label': dev0_label, 'posx': dev0_posx, 'posy': dev0_posy, 'posz': dev0_posz} 
    transceivers.append(transceiver1)

    device1 = d['device1']
    dev1_label = d['dev1_label']
    dev1_posx = d['dev1_posx']
    dev1_posy = d['dev1_posy']
    dev1_posz = d['dev1_posz']
    dev1_paths_pathfile = d['dev1_paths_pathfile']
    transceiver1 = {'type': device1, 'label': dev1_label, 'posx': dev1_posx, 'posy': dev1_posy, 'posz': dev1_posz} 
    transceivers.append(transceiver1)
    paths1 = np.load(dev1_paths_pathfile)
    print (np.shape(paths1))

    device2 = d['device2']
    dev2_label = d['dev2_label']
    dev2_posx = d['dev2_posx']
    dev2_posy = d['dev2_posy']
    dev2_posz = d['dev2_posz']
    dev2_paths_pathfile = d['dev2_paths_pathfile']
    transceiver2 = {'type': device2, 'label': dev2_label, 'posx': dev2_posx, 'posy': dev2_posy, 'posz': dev2_posz} 
    transceivers.append(transceiver2)

    device3 = d['device3']
    dev3_label = d['dev3_label']
    dev3_posx = d['dev3_posx']
    dev3_posy = d['dev3_posy']
    dev3_posz = d['dev3_posz']
    dev3_paths_pathfile = d['dev3_paths_pathfile']
    transceiver3 = {'type': device3, 'label': dev3_label, 'posx': dev3_posx, 'posy': dev3_posy, 'posz': dev3_posz} 
    transceivers.append(transceiver3)



    device4 = d['device4']
    dev4_label = d['dev4_label']
    dev4_posx = d['dev4_posx']
    dev4_posy = d['dev4_posy']
    dev4_posz = d['dev4_posz']
    dev4_paths_pathfile = d['dev4_paths_pathfile']
    transceiver4 = {'type': device4, 'label': dev4_label, 'posx': dev4_posx, 'posy': dev4_posy, 'posz': dev4_posz} 
    transceivers.append(transceiver4)


    device5 = d['device5']
    dev5_label = d['dev5_label']
    dev5_posx = d['dev5_posx']
    dev5_posy = d['dev5_posy']
    dev5_posz = d['dev5_posz']
    dev5_paths_pathfile = d['dev5_paths_pathfile']
    transceiver5 = {'type': device5, 'label': dev5_label, 'posx': dev5_posx, 'posy': dev5_posy, 'posz': dev5_posz} 
    transceivers.append(transceiver5)

    device6 = d['device6']
    dev6_label = d['dev6_label']
    dev6_posx = d['dev6_posx']
    dev6_posy = d['dev6_posy']
    dev6_posz = d['dev6_posz']
    dev6_paths_pathfile = d['dev6_paths_pathfile']
    transceiver6 = {'type': device6, 'label': dev6_label, 'posx': dev6_posx, 'posy': dev6_posy, 'posz': dev6_posz} 
    transceivers.append(transceiver6)

    device7 = d['device7']
    dev7_label = d['dev7_label']
    dev7_posx = d['dev7_posx']
    dev7_posy = d['dev7_posy']
    dev7_posz = d['dev7_posz']
    dev7_paths_pathfile = d['dev7_paths_pathfile']
    transceiver7 = {'type': device7, 'label': dev7_label, 'posx': dev7_posx, 'posy': dev7_posy, 'posz': dev7_posz} 
    transceivers.append(transceiver7)

    device8 = d['device8']
    dev8_label = d['dev8_label']
    dev8_posx = d['dev8_posx']
    dev8_posy = d['dev8_posy']
    dev8_posz = d['dev8_posz']
    dev8_paths_pathfile = d['dev8_paths_pathfile']
    transceiver8 = {'type': device8, 'label': dev8_label, 'posx': dev8_posx, 'posy': dev8_posy, 'posz': dev8_posz} 
    transceivers.append(transceiver8)

    device9 = d['device9']
    dev9_label = d['dev9_label']
    dev9_posx = d['dev9_posx']
    dev9_posy = d['dev9_posy']
    dev9_posz = d['dev9_posz']
    dev9_paths_pathfile = d['dev9_paths_pathfile']
    transceiver9 = {'type': device9, 'label': dev9_label, 'posx': dev9_posx, 'posy': dev9_posy, 'posz': dev9_posz} 
    transceivers.append(transceiver9)

    device10 = d['device10']
    dev10_label = d['dev10_label']
    dev10_posx = d['dev10_posx']
    dev10_posy = d['dev10_posy']
    dev10_posz = d['dev10_posz']
    dev10_paths_pathfile = d['dev10_paths_pathfile']
    transceiver10 = {'type': device10, 'label': dev10_label, 'posx': dev10_posx, 'posy': dev10_posy, 'posz': dev10_posz} 
    transceivers.append(transceiver10)

    #x = []
    #y = []
    #z = []
    #label = []
    #for transceiver in transceivers:
    #    print (transceiver)
    #    x.append(transceiver['posx'])
    #    y.append(transceiver['posy'])
    #    z.append(transceiver['posz'])
    #    label.append(transceiver['type']+transceiver['label'])
        
    #plot_devices(transceivers)

    x_min = 0
    x_max = 900
    
    y_min = 0
    y_max = 900

    z_min = 0
    z_max = 900
    n = norm(paths1[4])
    paths1[4] = paths1[4]/n
    print (n)
    #s = [1 for i in range(len(transceivers))]
    paths1 = paths1[:,0:1000]
    for p in paths1.T:
        aoa_theta = np.deg2rad(p[0])
        aoa_phi = np.deg2rad(p[1])
        r = p[4]
        x = r * np.sin(aoa_theta) * np.cos(aoa_phi)
        y = r * np.sin(aoa_theta) * np.sin(aoa_phi)
        z = r * np.cos(aoa_theta) 
        mlab.plot3d([0,x], [0,y], [0,z], tube_radius=0.000025)#, colormap="copper", scale_factor=5)
    mlab.axes(ranges=[x_min, x_max, y_min, y_max, z_min, z_max]) 
    #mlab.axes()
    #for i in range(len(label)):
    #    mlab.text3d(x[i]+5, y[i]+5, z[i]+5, label[i], scale=5)
    #mlab.view(azimuth=180, elevation=0)
    #mlab.view(azimuth=90, elevation=None)
    mlab.show()
