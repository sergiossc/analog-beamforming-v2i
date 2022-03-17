import sys
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    
    profile_pathfile = str(sys.argv[1])

    with open(profile_pathfile) as profile:
        data = profile.read()
        d = json.loads(data)

    prefix_pathfiles = d['results_directory']
    result_files = os.listdir(prefix_pathfiles)
    pathfiles = check_files(prefix_pathfiles, result_files)
    print ('# of json files: ', len(pathfiles))
    
    n = int(sys.argv[2])

    rx_array_size = n
    tx_array_size = n

    num_of_levels_available = [4, 8, 16, 32, 64, 128, 256, 512]

    count = 0
    min_distortion_info_dict = {}
    for n_levels in num_of_levels_available:
        min_distortion_info_dict[n_levels] = {'min_distortion':np.inf, 'result_pathfile': None}

    for pathfile_id, pathfile in pathfiles.items():
        with open(pathfile) as result:
            data = result.read()
            d = json.loads(data)

        # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
        if d['rx_array_size'] == rx_array_size and d['tx_array_size'] == tx_array_size: # and d['num_of_levels'] == num_of_levels:
            num_of_levels = d['num_of_levels']
            distortion_by_round = d['mean_distortion_by_round']
            mean_distortion = None
            for r, distortion in distortion_by_round.items():
                mean_distortion = dict2matrix(distortion)
            #min_distortion_dict[pathfile_id] = mean_distortion[-1]
            if mean_distortion[-1] < min_distortion_info_dict[num_of_levels]['min_distortion']:
                min_distortion_info_dict[num_of_levels]['min_distortion'] = mean_distortion[-1]
                min_distortion_info_dict[num_of_levels]['result_pathfile'] = pathfile
            count = count + 1
        #num_of_levels = d['num_of_levels']
    print (count)
    #print (f'min_distortion_pathfile, min_distortion: {min_distortion_pathfile, min_distortion}')
    print ("{")
    for k, v in min_distortion_info_dict.items():
        v_pathfile = v['result_pathfile']
        print (f'\t"{k}": "{v_pathfile}",')
    print ("}")
