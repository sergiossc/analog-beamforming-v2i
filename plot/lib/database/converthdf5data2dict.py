import pandas as pd
import numpy as np
import os
import uuid
import scipy.io as io
import h5py

def readepisodesfromhd5files(pathfile_id, pathfile):
    """
        Input: HDF5 files with data from RT. Episode is a set of Scenes. Scenes are a set of channel, and a scene is a set of paths. Each path has a set of information: received power, time of arrival, elevation, elevation_angle_of_departure, , azimuth_angle_of_departure , elevation_angle_of_arrival , azimuth_angle_of_arrival,  los ('1' is los and '0' is nlos).
        Output: Dict data read from HDF5. 
    """
    #print (pathfile)
    try:
        current_file = open(pathfile)
    except IOError:
        print('Cannot open')
    finally:
        current_file.close()

    f = h5py.File(pathfile, 'r')
    current_episode = f['allEpisodeData']
    
    #theta = np.linspace(0, (np.pi), 100)
    #phi = np.linspace(0, (2*np.pi), 100)
    count = 0
    episode_id = pathfile_id
    episode = {}
    scenes = {}
    scene_id = -1
    for scene in current_episode:
        #scene_id = uuid.uuid4()
        scene_id += 1 #uuid.uuid4()
        channels = {}
        channel_id = -1
        for channel in scene:
            #channel_id = uuid.uuid4()
            channel_id += 1 #uuid.uuid4()
            paths = {}
            path_id = -1
            for p in channel:
                if check_path_features(p):
                    #t = np.random.choice([45, 135], 1)
                    #path_id = uuid.uuid4()
                    path_id += 1 #uuid.uuid4()
                    path = {}
                    #print ('-----------------> p[0]9dBm: ', p[0])
                    path['received_power'] = np.power(10, (p[0]/10))  # convert data get in dBm to mW
                    ##path['received_power'] = np.power(10, (p[0]/10))/1000  # convert data get in dBm to W
                    #print (path['received_power']) # = np.power(10, (p[0]/10))/1000  # convert data get in dBm to W
                    path['time_of_arrival'] = p[1] # in seconds 
                    path['departure_theta'] = np.deg2rad(p[2]) # convert given angle from deg to rad same as before
                    #path['departure_theta'] = np.random.choice(theta) # np.deg2rad(p[2]) # convert given angle from deg to rad same as before
                    path['departure_phi'] =  np.deg2rad(p[3]) # same as before...
                    #path['departure_phi'] =  np.random.choice(phi) #np.deg2rad(p[3]) # same as before...
                    path['arrival_theta'] =  np.deg2rad(p[4]) # ...
                    #path['arrival_theta'] =  np.random.choice(theta) #np.deg2rad(p[4]) # ...
                    path['arrival_phi'] =  np.deg2rad(p[5]) # ...
                    #path['arrival_phi'] =  np.random.choice(phi) #np.deg2rad(p[5]) # ...
                    path['los'] = p[6] # '1' for a los, and '0' for a nlos path
                    ##path['phase'] = np.deg2rad(p[7]) # convert given phase from deg to rad 
                    ##print (f'type: {type(p)}')
                    ##print (f'len: {len(p)}')
                    paths[path_id] = path
                    #if p[6] > 0:
                    #    print ('los: ', p[6]) 
                    #else:
                    #    print ('nlos: ', p[6]) 

            if len(paths)>0:
                channels[channel_id] = paths
            else:
                count += 1
        #one_channel = {channels.popitem()[0]: channels.popitem()[1]}
        scenes[scene_id] = channels
        #scenes[scene_id] = one_channel

    episode[episode_id] = scenes
    #print ('==============================================================> count: ', count)

    return episode

def check_path_features(path):
    for feature in path:
        if np.isnan(feature):
            return False
        else:
            return True

#pathfile = '/home/snow/github/land/dataset/s002/ray_tracing_data_s002_carrier60GHz/rosslyn_fixed_60GHz_Ts3s_V_e1641.hdf5'
#path_id = 'c1608918-8796-4644-ba95-12857fbf2422'
#episode = readepisodesfromhd5files(path_id, pathfile)
##
#count_episodes = 0
#count_scenes = 0
#count_channels = 0
#
#for ep_id, scenes in episode.items():
#    print ('episode_id: ', ep_id)
#    count_episodes += 1
#    for scene_id, scene in scenes.items():
#        print ('scene_id: ', scene_id)
#        count_scenes += 1
#        for chn_id, channel in scene.items():
#            print (chn_id) 
#            count_channels += 1
#            for path_id, path in channel.items():
#                print ('path: ', path)
#
#print ('#episodes: ', count_episodes)
#print ('#scenes: ', count_scenes)
#print ('#channels: ', count_channels)
