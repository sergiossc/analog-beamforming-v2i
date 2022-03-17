import os
import numpy as np

#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/database')
#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/mimo')
#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/vq')
import load_lib
from database.preprocessing import check_hdf5files
from database.getsamples import get_paths_by_receiver

import database.dataconfig as datacfg
import mimo.arrayconfig as arraycfg

import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import pandas as pd
 
prefix_episodefiles = datacfg.prefix_episodefiles
dataset_name = datacfg.dataset_name

episode_files = os.listdir(prefix_episodefiles)
pathfiles = check_hdf5files(prefix_episodefiles, episode_files)
print (len(pathfiles))
path_set = get_paths_by_receiver(None, pathfiles)
    
#    rx = 5
for rx_id, paths in path_set.items():
    print (f'Receiver: RX{rx_id}')
    
    aoa_theta = []
    aoa_phi = []
    aod_theta = []
    aod_phi = []
    pwr = []

    for p in paths:
        aoa_theta.append(np.rad2deg(p['arrival_theta']))
        aoa_phi.append(np.rad2deg(p['arrival_phi']))
        aod_theta.append(np.rad2deg(p['departure_theta']))
        aod_phi.append(np.rad2deg(p['departure_phi']))
        pwr.append(p['received_power'])

    ##df = pd.DataFrame(u, index=dtheta, columns=dphi)
    data = np.array([aoa_theta, aoa_phi, aod_theta, aod_phi, pwr])
    data_filename = f'{dataset_name}_rx{rx_id}_paths'
    np.save(data_filename, data)
    print (f'filename: {data_filename}')
