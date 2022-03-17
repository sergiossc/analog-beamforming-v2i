#!/usr/bin/python3
import os
import numpy as np
import pandas as pd
#import seaborn as sns; sns.set_theme()
from numpy.linalg import norm, svd, matrix_rank
#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/database')
#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/mimo')
#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/vq')
import load_lib
#from database.preprocessing import check_hdf5files
#from database.getsamples import samplesfromhdf5files
from vq.utils import covariance_matrix #, norm
#import database.dataconfig as datacfg
import mimo.arrayconfig as arraycfg
from utils import plot_pattern, get_precoder_combiner, richscatteringchnmtx, loschnmtx #(num_rx, num_tx, variance):
from lib.utils.plot import PParttern
#from utils import plot_pattern, AntennaPartern, PlotPattern2, get_precoder_combiner, richscatteringchnmtx, loschnmtx #(num_rx, num_tx, variance):
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from mayavi import mlab
#from lib.utils.plot import plot_pattern, PlotPattern2

#num_rx = 16
#num_tx = 64
#variance = 1.0

#print (f'...')

my_seed = np.random.choice(10000)
print (my_seed)
np.random.seed(my_seed)

#num_of_samples = 1

#random_channels_set = []
#norm_value = []
#for i in range(num_of_samples):
#    h = richscatteringchnmtx(num_rx, num_tx, variance)
#    norm_value.append(norm(h))
#    random_channels_set.append(h/norm(h))
#random_channels_set = np.array(random_channels_set)

#h = richscatteringchnmtx(num_rx, num_tx, variance)
#h = np.matrix(h)

n = 4 # num of elements on rx
m = 4 # num of elements on tx

h_random_list = np.load(f'random_channels_set_{n}x{m}.npy')
h_los_rt_list = np.load(f'los_channels_{n}x{m}.npy')
h_nlos_rt_list = np.load(f'nlos_channels_{n}x{m}.npy')

n_random = np.random.choice(len(h_random_list))
h_random = np.matrix(h_random_list[n_random,:,:])
n_los = np.random.choice(len(h_los_rt_list))
h_los = np.matrix(h_los_rt_list[n_los,:,:])
n_nlos = np.random.choice(len(h_nlos_rt_list))
h_nlos = np.matrix(h_nlos_rt_list[n_nlos,:,:])

rx_array = arraycfg.rx_array
tx_array = arraycfg.tx_array
#aoa = np.random.choice(360)
aoa = -45 # deg
#aod = np.random.choice(360)
aod = 45 # deg
print (f'aoa: {aoa}')
print (f'aod: {aod}')
complex_gain = 1.0 * np.exp(1j * 0)
h_los_a = loschnmtx(complex_gain, rx_array, tx_array, aoa, aod)

h_random = 4 * h_random/norm(h_random)
h_los = 4 * h_los/norm(h_los)
h_nlos = 4 * h_nlos/norm(h_nlos)
h_los_a = 4 * h_los_a/norm(h_los_a)

precoder, combining = get_precoder_combiner(h_los_a)
#print (f'precoder: {np.rad2deg(np.angle(precoder))}')
#print (f'combining: {np.rad2deg(np.angle(combining))}')
#plot_pattern(tx_array, np.array(precoder))
#plot_pattern(rx_array, np.array(combining.T))
#PParttern(tx_array, precoder)
####PlotPattern2(tx_array)
####AntennaPartern(tx_array, np.array(precoder))
###
####print (f'norm: {norm(h_random)}')
####print (f'norm: {norm(h_los)}')
####print (f'norm and shape of h_nlos: {norm(h_nlos)} e {h_nlos.shape}')
####print (f'norm and rank of h_los_a: {norm(h_los_a)} and {matrix_rank(h_los_a)}')
###
#### SVD on channel samples
###u_random, s_random, vh_random = svd(h_random)
###u_los, s_los, vh_los = svd(h_los)
###u_nlos, s_nlos, vh_nlos = svd(h_nlos)
###u_los_a, s_los_a, vh_los_a = svd(h_los_a)
###
###
#### Eigenvalues from channel matrices
###corr_m_random = h_random.conj().T * h_random
###corr_n_random = h_random * h_random.conj().T
###
###corr_m_los = h_los.conj().T * h_los
###corr_n_los = h_los * h_los.conj().T
###
###corr_m_nlos = h_nlos.conj().T * h_nlos
###corr_n_nlos = h_nlos * h_nlos.conj().T
###
###corr_m_los_a = h_los_a.conj().T * h_los_a
###corr_n_los_a = h_los_a * h_los_a.conj().T
###
###
#### Plot effort -- Part1
###
###fig, (ax1, ax2) = plt.subplots(2, 1)
###ax1.plot(s_random ** 2, label='random')
###ax1.plot(s_los ** 2, label='los')
###ax1.plot(s_nlos ** 2, label='nlos')
###ax1.plot(s_los_a ** 2, label='los_a')
###ax1.set_title('eigenvalues dist')
###ax1.legend()
###
###ax2.plot(np.cumsum(s_random ** 2)/np.sum(s_random ** 2), label='random')
###ax2.plot(np.cumsum(s_los ** 2)/np.sum(s_los ** 2), label='los')
###ax2.plot(np.cumsum(s_nlos ** 2)/np.sum(s_nlos ** 2), label='nlos')
###ax2.plot(np.cumsum(s_los_a ** 2)/np.sum(s_los_a ** 2), label='los_a')
###ax2.set_title('eigenvalues cumsum')
###ax2.legend()
###
###plt.show()
###
#### Plot effort -- Part2
###
####fig, axes = plt.subplots(4, 2, sharex=True, sharey=True)
####
####df1 = pd.DataFrame(np.abs(corr_m_random), index=np.arange(m), columns=np.arange(m))
####sns.heatmap(df1, ax=axes[0,0], cmap='PiYG')
####df2 = pd.DataFrame(np.abs(corr_n_random), index=np.arange(n), columns=np.arange(n))
####sns.heatmap(df2, ax=axes[0,1], cmap='PiYG')
####
####
####df3 = pd.DataFrame(np.abs(corr_m_los), index=np.arange(m), columns=np.arange(m))
####sns.heatmap(df3, ax=axes[1,0], cmap='PiYG')
####df4 = pd.DataFrame(np.abs(corr_n_los), index=np.arange(n), columns=np.arange(n))
####sns.heatmap(df4, ax=axes[1,1], cmap='PiYG')
####
####
####df5 = pd.DataFrame(np.abs(corr_m_nlos), index=np.arange(m), columns=np.arange(m))
####sns.heatmap(df5, ax=axes[2,0], cmap='PiYG')
####df6 = pd.DataFrame(np.abs(corr_n_nlos), index=np.arange(n), columns=np.arange(n))
####sns.heatmap(df6, ax=axes[2,1], cmap='PiYG')
####
####df7 = pd.DataFrame(np.abs(h_los_a), index=np.arange(m), columns=np.arange(m))
####sns.heatmap(df7, ax=axes[3,0], cmap='PiYG')
####df8 = pd.DataFrame(np.abs(corr_n_los_a), index=np.arange(n), columns=np.arange(n))
####sns.heatmap(df8, ax=axes[3,1], cmap='PiYG')
####
####plt.show()
###