'''
https://www.mathworks.com/matlabcentral/fileexchange/27932-3d-array-factor-of-a-4x4-planar-array-antenna
'''
import sys
sys.path.append(r'/home/snow/github/land/lib/mimo_tools/')
sys.path.append(r'/home/snow/github/land/lib/mimo_tools/utils')
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from scipy.constants import c
from mpl_toolkits import mplot3d
from numpy.linalg import norm
from phased import PartitionedArray
from plot import plot_pattern, plot_cb_pattern
from switched_codebook import dft_codebook



fc = 60 * (10 ** 9)
wavelength = c/fc
d =  wavelength/2
 
# Uniform Planar Array Setup
num_tx = 16
tx_array = PartitionedArray(num_tx, d, wavelength, "UPA") # configuracao da antena de transmissao
#(self, size, element_spacing, num_rf, wave_length, formfactory):
cb = dft_codebook(tx_array)
for cw_id, cw in cb.items():
    print (cw_id)
    unitary_cb = {}
    unitary_cb[cw_id] = cw
    plot_cb_pattern(tx_array, unitary_cb)
# w
#plot_pattern(tx_array, cd)
#plot_cb_pattern(tx_array, cb)
