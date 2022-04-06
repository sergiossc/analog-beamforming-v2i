import sys
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    
    #profile_pathfile = 'profile-rt-s002.json' 
    #profile_pathfile = str(sys.argv[1])

    #with open(profile_pathfile) as profile:
    #    data = profile.read()
    #    d = json.loads(data)

    #prefix_pathfiles = d['results_directory']
    prefix_pathfiles = "/home/snow/analog-beamforming-v2i/results/s002"
    result_files = os.listdir(prefix_pathfiles)
    pathfiles = check_files(prefix_pathfiles, result_files)
    print ('# of json files: ', len(pathfiles))
    #num_of_levels_opts = d['num_of_levels_opts']

    #gridsize = (2, 3)
    #fig = plt.figure(figsize=(8, 12))
    
    initial_alphabet_opt = str(sys.argv[1])
    n = int(sys.argv[2])
    #n = 32

    rx_array_size = n
    tx_array_size = n

    #initial_alphabet_opt = str(sys.argv[3])
    #initial_alphabet_opt = 'xiaoxiao' #str(sys.argv[3])
    #initial_alphabet_opt = 'random' #str(sys.argv[3])
    #num_of_levels = int(sys.argv[2])
    num_of_levels = n

    count = 0

    distortion_data = {}
    for pathfile_id, pathfile in pathfiles.items():
        with open(pathfile) as result:
            data = result.read()
            d = json.loads(data)

        # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
        #if d['rx_array_size'] == rx_array_size and d['tx_array_size'] == tx_array_size and d['initial_alphabet_opt'] == initial_alphabet_opt:
        #if d['num_of_levels'] == num_of_levels:
        if d['initial_alphabet_opt'] == initial_alphabet_opt and d['tx_array_size'] == tx_array_size and d['rx_array_size'] == rx_array_size and d['num_of_levels'] == num_of_levels:
        #if d['rx_array_size'] == rx_array_size and d['tx_array_size'] == tx_array_size:
        #if d['channel_samples_filename'] == "/home/snow/analog-beamforming-v2i/s000-training_set_4x4_a.npy":
            #if d['initial_alphabet_opt'] == initial_alphabet_opt:
            count = count + 1
            print (f'pathfile: {pathfile}')
            #print (f"num of levels: {d['num_of_levels']}\n")
        #num_of_levels = d['num_of_levels']
    print (count)
