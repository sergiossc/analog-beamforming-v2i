import sys

#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/utils')
#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/mimo')

import database.dataconfig as datacfg
import pandas as pd
from mimo.channel import scatteringchnmtxfromtextfiles, scatteringchnmtxfromhdf5files, richscatteringchnmtx
import uuid
import numpy as np
from database.converttextdata2dict import readtransceiversfromtextfiles, readpathsfromtextfiles
from database.converthdf5data2dict import readepisodesfromhd5files


def randomsamples(num_samples, tx_array, rx_array):
    sample_set = {}
    for s in range(num_samples):
        h = richscatteringchnmtx(tx_array.size, rx_array.size)
        sample_id = uuid.uuid4()
        sample_set_set[sample_id] = h
    return sample_set

def samplesfromhdf5files(csvsetfile, rx_array, tx_array, pathfiles=None):
    sample_set = {}
    num_of_valid_channels = 0

    if csvsetfile is not None:
        df = pd.read_csv(csvsetfile, index_col=[0])
        pathfiles = df.to_dict()
    for pathfile_id, pathfile in pathfiles.items():
        #pass
        #print (f'pathfile: {pathfile}')
        #try:
        #    current_file = open(pathfile)
        #    print (f'Sucess')
        #except IOError:
        #    print('Cannot open')
        #finally:
        #    current_file.close()
        #path_episode = pathfile[0]
        path_episode = pathfile
        episode = readepisodesfromhd5files(pathfile_id, path_episode)
        for ep_id, scenes in episode.items():
            #print ('episode id: ', ep_id)
            #scenes_id = scenes.keys()
            #scenes = np.random.choice(scenes, 1)
            for scene_id, scene in scenes.items():
                #print (scene_id)
                for chn_id, channel in scene.items():
                    #print (f'chn_id: {chn_id}') 
                    paths = []
                    for path_id, path in channel.items():
                        paths.append(path)
                    if len(paths) > 0: # this is a valid channel there is at least one ray
                        num_of_valid_channels += 1
                        #plot_paths(path_scene, chn_paths, tx_transceiver, rx_transceiver)
                        sample_id = uuid.uuid4()
                        #print ('len(chn_paths): ', len(chn_paths))
                        h = scatteringchnmtxfromhdf5files(paths, rx_array, tx_array)
                        is_there_los_paths = False
                        #PParttern(path_scene, h, tx_array)
                        for p in paths:
                            if p['los'] > 0.0:
                                is_there_los_paths = True
                                #print (type(p['los']))
                                #print (p['los'])
                        sample_set[sample_id] = {'channel_matrix': h, 'ep_id': ep_id, 'is_there_los_paths': is_there_los_paths}
#
    print (f'%num_of_valid_channels: {num_of_valid_channels}')
    return sample_set
