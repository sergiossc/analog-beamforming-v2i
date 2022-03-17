import os
import numpy as np

#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/database')
#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/mimo')
#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/vq')
import load_lib
from database.preprocessing import check_hdf5files
from database.getsamples import samplesfromhdf5files
from vq.utils import covariance_matrix, norm

import database.dataconfig as datacfg
import mimo.arrayconfig as arraycfg

rx_array = arraycfg.rx_array
tx_array = arraycfg.tx_array
 
kfold_number = datacfg.kfold_number
prefix_episodefiles = datacfg.prefix_episodefiles
data_to_use = datacfg.data_to_use
create_validation_set = datacfg.create_validation_set

episode_files = os.listdir(prefix_episodefiles)
pathfiles = check_hdf5files(prefix_episodefiles, episode_files)

dict_channels = samplesfromhdf5files(None, rx_array, tx_array, pathfiles)

print (f'%# of episode_files: {len(episode_files)}')

channels_keys = [k for k in dict_channels.keys()]

rx1_nlos_channels = []
rx2_nlos_channels = []
rx3_nlos_channels = []
rx4_nlos_channels = []
rx5_nlos_channels = []
rx6_nlos_channels = []
rx7_nlos_channels = []
rx8_nlos_channels = []
rx9_nlos_channels = []
rx10_nlos_channels = []

for k, v in dict_channels.items():
    h = v['channel_matrix']
    h = arraycfg.num_tx * (h/norm(h)) #normatize channel sample 
    #print (f'norm(h): {norm(h)}')
    rx_id = int(v['receiver_id']) #receiver_id
    print (rx_id)
    if rx_id == int(1):
        rx1_nlos_channels.append(h)
    if rx_id == int(2):
        rx2_nlos_channels.append(h)
    if rx_id == int(3):
        rx3_nlos_channels.append(h)
    if rx_id == int(4):
        rx4_nlos_channels.append(h)
    if rx_id == int(5):
        rx5_nlos_channels.append(h)
    if rx_id == int(6):
        rx6_nlos_channels.append(h)
    if rx_id == int(7):
        rx7_nlos_channels.append(h)
    if rx_id == int(8):
        rx8_nlos_channels.append(h)
    if rx_id == int(9):
        rx9_nlos_channels.append(h)
    if rx_id == int(10):
        rx10_nlos_channels.append(h)



rx1_nlos_channels = np.array(rx1_nlos_channels)
rx2_nlos_channels = np.array(rx2_nlos_channels)
rx3_nlos_channels = np.array(rx3_nlos_channels)
rx4_nlos_channels = np.array(rx4_nlos_channels)
rx5_nlos_channels = np.array(rx5_nlos_channels)
rx6_nlos_channels = np.array(rx6_nlos_channels)
rx7_nlos_channels = np.array(rx7_nlos_channels)
rx8_nlos_channels = np.array(rx8_nlos_channels)
rx9_nlos_channels = np.array(rx9_nlos_channels)
rx10_nlos_channels = np.array(rx10_nlos_channels)

rx1_nlos_channels_filename = f's002_rx1_nlos_channels_{rx_array.size}x{tx_array.size}'
rx2_nlos_channels_filename = f's002_rx2_nlos_channels_{rx_array.size}x{tx_array.size}'
rx3_nlos_channels_filename = f's002_rx3_nlos_channels_{rx_array.size}x{tx_array.size}'
rx4_nlos_channels_filename = f's002_rx4_nlos_channels_{rx_array.size}x{tx_array.size}'
rx5_nlos_channels_filename = f's002_rx5_nlos_channels_{rx_array.size}x{tx_array.size}'
rx6_nlos_channels_filename = f's002_rx6_nlos_channels_{rx_array.size}x{tx_array.size}'
rx7_nlos_channels_filename = f's002_rx7_nlos_channels_{rx_array.size}x{tx_array.size}'
rx8_nlos_channels_filename = f's002_rx8_nlos_channels_{rx_array.size}x{tx_array.size}'
rx9_nlos_channels_filename = f's002_rx9_nlos_channels_{rx_array.size}x{tx_array.size}'
rx10_nlos_channels_filename = f's002_rx10_nlos_channels_{rx_array.size}x{tx_array.size}'

np.save(rx1_nlos_channels_filename, rx1_nlos_channels)
np.save(rx2_nlos_channels_filename, rx2_nlos_channels)
np.save(rx3_nlos_channels_filename, rx3_nlos_channels)
np.save(rx4_nlos_channels_filename, rx4_nlos_channels)
np.save(rx5_nlos_channels_filename, rx5_nlos_channels)
np.save(rx6_nlos_channels_filename, rx6_nlos_channels)
np.save(rx7_nlos_channels_filename, rx7_nlos_channels)
np.save(rx8_nlos_channels_filename, rx8_nlos_channels)
np.save(rx9_nlos_channels_filename, rx9_nlos_channels)
np.save(rx10_nlos_channels_filename, rx10_nlos_channels)
print(f'done!')
