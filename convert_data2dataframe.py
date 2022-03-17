import os
import numpy as np
import pandas as pd
import load_lib
from database.preprocessing import check_hdf5files
from database.converthdf5data2dict import readepisodesfromhd5files
import database.dataconfig as datacfg

prefix_episodefiles = datacfg.prefix_episodefiles
episode_files = os.listdir(prefix_episodefiles)
pathfiles = check_hdf5files(prefix_episodefiles, episode_files)

csv_position_file = datacfg.csv_position_file
pos_info = pd.read_csv(csv_position_file)
pos_info = pos_info[['EpisodeID', 'SceneID', 'VehicleArrayID', 'x', 'y', 'z']]
print (pos_info.head(30))

eps_id = []
scenes_id = []
channels_id = []
x = []
y = []
z = []
paths_id = []
received_power_vals = []
time_of_arrival_vals = []
departure_theta_vals = []
departure_phi_vals = []
arrival_theta_vals = []
arrival_phi_vals = []
los_vals = []

#path: {'received_power': 1.026362224093333e-18, 'time_of_arrival': 1.09571e-06, 'departure_theta': 1.5707964, 'departure_phi': 2.2445805, 'arrival_theta': 1.4010054, 'arrival_phi': 2.7044225, 'los': 0.0}


for ep_id, ep_pathfile in pathfiles.items():
    ep = readepisodesfromhd5files(ep_id, ep_pathfile)
    #print (f'ep_id: {ep_id}')
    scenes = ep[ep_id]
    for scene_id, scene in scenes.items():
        #print (f'scene_id: {scene_id}')
        for channel_id, channel in scene.items():
            #print (f'channel_id: {channel_id}')
            for path_id, path in channel.items():
                #for k, v in path.items():
                eps_id.append(ep_id)
                scenes_id.append(scene_id)
                channels_id.append(channel_id)
                #xyz = pos_info[(pos_info['EpisodeID'] == ep_id) & ('SceneID' == scene_id) & ('VehicleArrayID' == channel_id)]
                xyz = pos_info[(pos_info['EpisodeID'] == ep_id) & (pos_info['SceneID'] == scene_id) & (pos_info['VehicleArrayID'] == channel_id)]
                xyz = pos_info[['x', 'y', 'z']]
                xyz = xyz.to_numpy()
                x.append(xyz[0,0]) #pos.append((xyz[0,0], xyz[0,1], xyz[0,2]))
                y.append(xyz[0,1]) #pos.append((xyz[0,0], xyz[0,1], xyz[0,2]))
                z.append(xyz[0,2]) #pos.append((xyz[0,0], xyz[0,1], xyz[0,2]))
                ##print (xyz)
                paths_id.append(path_id)
                received_power_vals.append(path['received_power'])
                time_of_arrival_vals.append(path['time_of_arrival'])
                departure_theta_vals.append(path['departure_theta'])
                departure_phi_vals.append(path['departure_phi'])
                arrival_theta_vals.append(path['arrival_theta'])
                arrival_phi_vals.append(path['arrival_phi'])
                los_vals.append(path['los'])

data = {'episode': eps_id, 'scene': scenes_id, 'channel': channels_id, 'x': x, 'y': y, 'z': z, 'path': paths_id, 'received_power': received_power_vals,'time_of_arrival': time_of_arrival_vals, 'departure_theta':departure_theta_vals, 'departure_phi': departure_phi_vals, 'arrival_theta': arrival_theta_vals, 'arrival_phi': arrival_phi_vals, 'los': los_vals}

df = pd.DataFrame(data)
df.to_csv('all_paths_data_from_007.csv')
#only_los_paths = pos_info[(data['los'] == 1)]
#print (only_los_paths.head())

