import sys
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import seaborn as sns

if __name__ == '__main__':

    #profile_pathfile = 'profile-rt-s004.json' 
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

    count = 0

    title = f"{d['channel_samples_files'][0]}" + " N=" + f"{n}"

    data_dict = {}
    for pathfile_id, pathfile in pathfiles.items():
        with open(pathfile) as result:
            data = result.read()
            d = json.loads(data)

        # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
        if d['rx_array_size'] == rx_array_size and d['tx_array_size'] == tx_array_size: # and d['initial_alphabet_opt'] == initial_alphabet_opt:
            count = count + 1
            print (d['channel_samples_filename'])

            num_of_levels = d['num_of_levels']
            initial_alphabet_opt = d['initial_alphabet_opt']
            data_dict[num_of_levels] = {} 

    print (data_dict)

    for pathfile_id, pathfile in pathfiles.items():
        with open(pathfile) as result:
            data = result.read()
            d = json.loads(data)
    
        # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
        if d['rx_array_size'] == rx_array_size and d['tx_array_size'] == tx_array_size: # and d['initial_alphabet_opt'] == initial_alphabet_opt:
            count = count + 1
    
            num_of_levels = d['num_of_levels']

            distortion_by_round = d['mean_distortion_by_round'] 
            mean_distortion = None
            for r, mean_distortion in distortion_by_round.items():
                mean_distortion = dict2matrix(mean_distortion)

            initial_alphabet_opt = d['initial_alphabet_opt']
            data_dict[num_of_levels][initial_alphabet_opt] = mean_distortion[-1]

    data_keys_ordered = sorted(data_dict) #, key=lambda k: k['num_of_levels'])
    data_new = {}
    for k in data_keys_ordered:
        data_new[k] = data_dict[k]
    print (data_new)

    color_dict = {'random': 'green', 'random_from_samples': 'red', 'xiaoxiao': 'orange', 'katsavounidis': 'blue' }
    df = pd.DataFrame(data_new).T
    sorted_cols = sorted(df.columns.tolist())
    df = df[sorted_cols]
    df.plot.bar(color = color_dict)
    print (df)
    print (title)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()
