'''
https://www.mathworks.com/matlabcentral/fileexchange/27932-3d-array-factor-of-a-4x4-planar-array-antenna
'''
import sys
import load_lib
#sys.path.append(r'/home/snow/github/land/lib/mimo_tools/')
#sys.path.append(r'/home/snow/github/land/lib/mimo_tools/utils')
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from scipy.constants import c
from mpl_toolkits import mplot3d
from numpy.linalg import norm
#from mimo.phased import PartitionedArray
from lib.utils.plot import plot_pattern, plot_cb_pattern
from lib.utils.switched_codebook import dft_codebook
from utils import norm, decode_codebook
import mimo.arrayconfig as arraycfg
import json

if __name__ == "__main__":

    samples_npy_file = sys.argv[1]
    samples = np.load(samples_npy_file)
    rx_array = arraycfg.rx_array
    tx_array = arraycfg.tx_array
    n_rx = rx_array.size
    n_tx = tx_array.size
    m = n_tx

    #with open(result_pathfile) as result:
    #    data = result.read()
    #    d = json.loads(data)

    #codebook_json = d['codebook']
    #cb_dict = decode_codebook(codebook_json)
    #cb = {}
    random_seed = np.random.choice(10000, 1)
    print (f'random_seed: {random_seed}')
    np.random.seed(random_seed)
    sample_id_list = np.random.choice(len(samples), 5, replace=False)
    cb = {}
    for sample_id in sample_id_list:
        print (sample_id)
        sample = samples[sample_id]
        print (f'norm(sample): \n{norm(sample)}')
        u, s, vh = svd(sample)
        f = vh[0,:]
        #w = u[:,0]
        #f = 1/np.sqrt(25) * np.exp(1j * np.deg2rad(30) * np.arange(25))
        cb[sample_id] = f
        print (f'f: \n{f}')
        print (f'norm(f): \n{norm(f)}')
        
        

#    for k, cw in cb_dict.items():
#        cw = np.array(cw).reshape(n_rx, n_tx)
#        #cw = m * cw/norm(cw)
#        print (f'norm(cw): {norm(cw)}')
#        u_s, s_s, vh_s = svd(cw)
#        u_s = np.matrix(u_s)
#        vh_s = np.matrix(vh_s)
#        f_s = vh_s[0,:]
#        w_s = u_s[:,0]
#        cb[k] = f_s
#        print (f'f_s: \n{f_s}')
#        
# 
    for cw_id, cw in cb.items():
        print (cw_id)
        print (f'shape of cw: {np.shape(cw)}')
        cw = np.array(cw).reshape(int(np.sqrt(n_tx)),int(np.sqrt(n_tx)))
        print (f'shape of cw: {np.shape(cw)}')
        print (f'norm of cw: {norm(cw)}')
        #cw = cw/norm(cw)
        print (f'norm of cw: {norm(cw)}')
        unitary_cb = {}
        unitary_cb[cw_id] = cw
        print (f'cw: \n{cw}')
        plot_cb_pattern(tx_array, unitary_cb)
#    # w
#    #plot_pattern(tx_array, cd)
#    #plot_cb_pattern(tx_array, cb)
