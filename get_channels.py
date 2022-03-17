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

from utils import array_to_angular_domain

rx_array = arraycfg.rx_array
tx_array = arraycfg.tx_array
 
kfold_number = datacfg.kfold_number
prefix_episodefiles = datacfg.prefix_episodefiles
data_to_use = datacfg.data_to_use
create_validation_set = datacfg.create_validation_set
get_samples_by_user = datacfg.get_samples_by_user # for num mobile rx

episode_files = os.listdir(prefix_episodefiles)
pathfiles = check_hdf5files(prefix_episodefiles, episode_files)

dict_channels = samplesfromhdf5files(None, rx_array, tx_array, pathfiles)

#  If angularize is True, get samples in angular domain
if datacfg.angularize:
    nr = arraycfg.num_rx
    nt = arraycfg.num_tx

    #d = arraycfg.element_spacing

    #wavelength = arraycfg.wave_length

    #s_rx = get_orthonormal_basis(nr, d, wavelength)
    #s_tx = get_orthonormal_basis(nt, d, wavelength)

    for k, ch in dict_channels.items():
        h = ch['channel_matrix']
        #h_a = s_rx.conj().T * (h * s_tx.conj())
        #h_a = nt * h_a/norm(h_a)
        h_a = array_to_angular_domain(h)
        h_a = np.sqrt(nr * nt) * h_a/norm(h_a)
        ch['channel_matrix'] = h_a

print (f'%# of episode_files: {len(episode_files)}')

print (f'%Getting k-fold sets(training, validation, and test), with k = {kfold_number}')
channels_keys = [k for k in dict_channels.keys()]

kfold_slot_length = int(len(channels_keys)/kfold_number)

#validation k-fold set
validation_channels_keys = np.random.choice(channels_keys, kfold_slot_length, replace=False)
validation_channels = {}
for k in validation_channels_keys:
    validation_channels[k] = dict_channels.pop(k)

#test k-fold set
channels_keys = [k for k in dict_channels.keys()]
test_channels_keys = np.random.choice(channels_keys, kfold_slot_length, replace=False)
test_channels = {}
for k in test_channels_keys:
    test_channels[k] = dict_channels.pop(k)

#training k-fold set
training_channels = dict_channels

print (f'...')

validation_set = []
validation_set_dict = {}

for k, v in validation_channels.items():

    if get_samples_by_user:
        h = v['channel_matrix']
        receiver_id = v['receiver_id']
        validation_set_dict[receiver_id] = []
        validation_set.append({receiver_id: h})
    else:
        h = v['channel_matrix']
        validation_set.append(h)

if get_samples_by_user:
    pass
    for dict_v in validation_set:
        pass
        for k, v in dict_v.items():
            receiver_id = k
            h = v
            validation_set_dict[receiver_id].append(h)
else:
    validation_set = np.array(validation_set)

test_set = []
test_set_dict = {}

for k, v in test_channels.items():

    if get_samples_by_user:
        h = v['channel_matrix']
        receiver_id =  v['receiver_id']
        test_set_dict[receiver_id] = []
        test_set.append({receiver_id: h})
    else:
        h = v['channel_matrix']
        test_set.append(h)

if get_samples_by_user:
    pass
    for dict_v in test_set:
        pass
        for k, v in dict_v.items():
            receiver_id = k
            h = v
            test_set_dict[receiver_id].append(h)
else:
    test_set = np.array(test_set)


training_set = []
training_set_dict = {}
for k, v in training_channels.items():

    if get_samples_by_user:
        h = v['channel_matrix']
        receiver_id =  v['receiver_id']
        training_set_dict[receiver_id] = []
        training_set.append({receiver_id: h})
    else:
        h = v['channel_matrix']
        training_set.append(h)

if get_samples_by_user:
    pass
    for dict_v in training_set:
        pass
        for k, v in dict_v.items():
            receiver_id = k
            h = v
            training_set_dict[receiver_id].append(h)
else:
    training_set = np.array(training_set)




if get_samples_by_user:
    pass
else:
    print (f'%training_set.shape = {training_set.shape}')
    print (f'%validation_set.shape = {validation_set.shape}')
    print (f'%test_set.shape = {test_set.shape}')

angular_str = ''
if datacfg.angularize:
    angular_str = '_a'

if get_samples_by_user:
    for k, v in training_set_dict.items():
        training_set_filename = f'{datacfg.dataset_name}-rx{k}-training_set_{rx_array.size}x{tx_array.size}{angular_str}'
        np.save(training_set_filename, np.array(v))
    for k, v in validation_set_dict.items():
        validation_set_filename = f'{datacfg.dataset_name}-rx{k}-validation_set_{rx_array.size}x{tx_array.size}{angular_str}'
        np.save(validation_set_filename, np.array(v))
    for k, v in test_set_dict.items():
        test_set_filename = f'{datacfg.dataset_name}-rx{k}-test_set_{rx_array.size}x{tx_array.size}{angular_str}'
        np.save(test_set_filename, np.array(v))
else:
    tranining_set_filename = f'{datacfg.dataset_name}-training_set_{rx_array.size}x{tx_array.size}{angular_str}'
    validation_set_filename = f'{datacfg.dataset_name}-validation_set_{rx_array.size}x{tx_array.size}{angular_str}'
    test_set_filename = f'{datacfg.dataset_name}-test_set_{rx_array.size}x{tx_array.size}{angular_str}'

    np.save(tranining_set_filename, training_set)
    np.save(validation_set_filename, validation_set)
    np.save(test_set_filename, test_set)

#print (f'%Data saved in \'{tranining_set_filename}.npy\', \'{validation_set_filename}.npy\', \'{test_set_filename}.npy\' files.')
