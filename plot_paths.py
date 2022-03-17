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
    pass
