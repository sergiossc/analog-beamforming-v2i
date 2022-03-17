from math import exp
import os
import numpy as np
from numpy.linalg import svd, matrix_rank, norm

#from lib.utils.plot import plot_pattern, plot_cb_pattern
#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/database')
#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/mimo')
#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/vq')
import load_lib
#from database.preprocessing import check_hdf5files
#from database.getsamples import samplesfromhdf5files
#from vq.utils import covariance_matrix, norm

#import database.dataconfig as datacfg
import mimo.arrayconfig as arraycfg
from utils import covariance_matrix, loschnmtx
np.set_printoptions(suppress=True)

rx_array = arraycfg.rx_array
tx_array = arraycfg.tx_array

print (rx_array.size)
print (tx_array.size)

#h = channel_mtx(paths, rx_array, tx_array)
aoa = 30
aod = 60
h = loschnmtx(rx_array, tx_array, aoa, aod) #channel_mtx(paths, rx_array, tx_array)
h = np.sqrt(tx_array.size) * h/norm(h)
h = np.matrix(h)
print (f'h: \n{h}')
print (f'rank(h): {matrix_rank(h)}')
u, s, vh = svd(h)
print (s[0] ** 2)

precoder = np.matrix(vh).conj().T[:,0]
combining = np.matrix(u).conj().T[0,:]

#print (f'precoder.shape: {precoder.shape}')
#print (f'precoder: {precoder}')
#print (f'combining.shape: {combining.shape}')
#print (f'combining: {combining}')
#print (f'norm(precoder): {norm(precoder)}') 
#print (f'norm(combining): {norm(combining)}') 

prod1 = h * precoder
prod2 = combining * prod1

#print (f'prod2: {prod2}')
#print (f'prod2.shape: {prod2.shape}')
#print (f'abs(prod2) ** 2: {np.abs(prod2[0,0]) ** 2}')

#print (f'abs(precoder): \n{np.abs(precoder)}')
#print (f'angle(precoder) deg: \n{np.rad2deg(np.angle(precoder))}')
#print (f'abs(combining): \n{np.abs(combining)}')
#print (f'angle(combining) deg: \n{np.rad2deg(np.angle(combining))}')
