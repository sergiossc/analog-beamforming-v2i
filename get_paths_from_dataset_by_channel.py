import os
import numpy as np
import load_lib
from database.preprocessing import check_hdf5files
from database.getsamples import get_paths_by_receiver_by_channel
import database.dataconfig as datacfg
import mimo.arrayconfig as arraycfg
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import pandas as pd
 
prefix_episodefiles = datacfg.prefix_episodefiles
dataset_name = datacfg.dataset_name

episode_files = os.listdir(prefix_episodefiles)
print (type(episode_files))
pathfiles = check_hdf5files(prefix_episodefiles, episode_files)
print (len(pathfiles))

#paths by channel
channels_path_set = get_paths_by_receiver_by_channel(None, pathfiles)

paths_set_by_receiver = {}

for k, v in channels_path_set.items():
    receiver_id = v['receiver_id']
    print (type(receiver_id))
    paths_set_by_receiver[receiver_id] = [] #.append(v['paths'])

print (paths_set_by_receiver.keys())       
print (paths_set_by_receiver.values())       

print (f'++++++++++++++++++++++++++++++')

for k, v in channels_path_set.items():
    receiver_id = v['receiver_id']
    paths_set_by_receiver[receiver_id].append(v['paths'])

print (f'****************************')
print (paths_set_by_receiver.keys())       


for receiver_id, channels in paths_set_by_receiver.items():
    print (f'receiver id: {receiver_id}')
    filename = dataset_name + f'_rx' + str(receiver_id) + f'_paths_by_channel'
    print (f'# of channels: {len(channels)}')
    paths2save = []
    for channel in channels:
        vec_p = [] 
        for p in channel:
            vec_p.append([p['arrival_theta'], p['arrival_phi'], p['departure_theta'], p['departure_phi'], p['received_power']])
        paths2save.append(np.array(vec_p))
    print (np.shape(paths2save))
    np.save(filename, paths2save)
print ('++++++++++++++++++++++++++++++++++++++++++++++')

    

#for receiver_id, channel in paths_set_by_receiver.items():
#    print (f'receiver_id: {receiver_id}, len(channel): {len(channel)}')
#    filename = dataset_name + f'_rx' + str(receiver_id) + f'_paths_by_channel'
#    print (filename)
#    paths2save = [] # set paths separeted by channel occurences
#    for paths in channel: # paths in a given pathset
#        my_p = []
#        for p in paths:
#            my_p.append([p['arrival_theta'], p['arrival_phi'], p['departure_theta'], p['departure_phi'], p['received_power']])
#        paths2save.append(my_p)
#    print (np.shape(paths2save))
