import sys
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import seaborn as sns
import uuid



if __name__ == '__main__':
    #n_set = [4, 8, 16, 32, 64]
    n_set = [4, 8, 16, 32, 64]
    setup_dict = {4:'I', 8:'II', 16:'III', 32:'IV', 64:'V'}

    initial_alphabet_set = ['katsavounidis', 'xiaoxiao', 'random_from_samples', 'random']
    num_levels_set = [4, 8, 16, 32, 64, 128, 256, 512]
    dataset_name = 's000'
    for n in n_set:
        user_filter_set = {}
        for num_levels in num_levels_set:
            for initial_alphabet in initial_alphabet_set:
                user_filter_id = uuid.uuid4()
                user_filter = {'ds_name': dataset_name, 'rx_array_size': n, 'tx_array_size': n, 'initial_alphabet_opt': initial_alphabet, 'num_of_levels': num_levels}
                user_filter_set[user_filter_id] = user_filter

        mean_distortion_dict = {}
        for filter_id in user_filter_set.keys():
            mean_distortion_dict[filter_id] = None
    
        result_pathfiles_dict = get_all_result_json_pathfiles()
    
        # Now I have to deal with each JSON result data file
        for k, pathfile in result_pathfiles_dict.items():
            with open(pathfile) as result:
                data = result.read()
                d = json.loads(data)
            pass
            ds_name = get_datasetname_from_json_resultfiles(d)
            rx_array_size = d['rx_array_size']
            tx_array_size = d['tx_array_size']
            initial_alphabet = d['initial_alphabet_opt']
            num_of_levels = d['num_of_levels']
            #channel_samples_filename = d['channel_samples_filename']
     
            result_filter = {'ds_name': ds_name, 'rx_array_size': rx_array_size, 'tx_array_size': tx_array_size, 'initial_alphabet_opt': initial_alphabet, 'num_of_levels': num_of_levels}
    
            for filter_id, user_filter in user_filter_set.items():
                if compare_filter(user_filter, result_filter):
                    pass
                    # From here handle matched filters data
                    print (f'user filter is {user_filter}')
                    print (f'result filter is {result_filter}\n')
                    print (f'{pathfile}') 
    
                    #n_levels = d['num_of_levels']
                    distortion_by_round = d['mean_distortion_by_round']
                    for r, mean_distortion in distortion_by_round.items():
                        mean_distortion_dict[filter_id] = dict2matrix(mean_distortion)
    
    

        color_dict = {'random': 'green', 'random_from_samples': 'red', 'xiaoxiao': 'orange', 'katsavounidis': 'blue' }
        marker_dict = {4: "*", 8: "h", 16: "p", 32: 4, 64:5, 128:6, 256:7, 512:10}
        fig, ax = plt.subplots()
        for filter_id, mean_distortion in mean_distortion_dict.items():
            pass
            ax.plot(10 * np.log10(mean_distortion), marker=marker_dict[user_filter_set[filter_id]['num_of_levels']],  color=color_dict[user_filter_set[filter_id]['initial_alphabet_opt']], label=f"{user_filter_set[filter_id]['initial_alphabet_opt']}, L={user_filter_set[filter_id]['num_of_levels']}")
        plt.legend(loc='best')
        plt.xlabel(f'Número de iterações')
        plt.ylabel(r'Valor esperado da distorção ($\mathbb{E}(D)$)') 
        plt.title(f'Experimento {setup_dict[n]} (N={n})')

        ax_small = fig.add_axes([0.3, 0.45, 0.3, 0.4])
        for filter_id, mean_distortion in mean_distortion_dict.items():
            pass
            ax_small.plot(10 * np.log10(mean_distortion), marker=marker_dict[user_filter_set[filter_id]['num_of_levels']],  color=color_dict[user_filter_set[filter_id]['initial_alphabet_opt']])
        ax_small.set_xlim(-1,10)
        ax_small.set_ylim(-45,2)
        plt.show()
#


#if __name__ == '__main__':
#
#    #profile_pathfile = 'profile-rt-s004.json' 
#    profile_pathfile = str(sys.argv[1])
#    with open(profile_pathfile) as profile:
#        data = profile.read()
#        d = json.loads(data)
#
#    prefix_pathfiles = d['results_directory']
#    result_files = os.listdir(prefix_pathfiles)
#    pathfiles = check_files(prefix_pathfiles, result_files)
#    print ('# of json files: ', len(pathfiles))
#    
#    n = int(sys.argv[2])
#
#    rx_array_size = n
#    tx_array_size = n
#
#    count = 0
#
#    title = f"{d['channel_samples_files'][0]}" + " N=" + f"{n}"
#
#    data_dict = {}
#    for pathfile_id, pathfile in pathfiles.items():
#        with open(pathfile) as result:
#            data = result.read()
#            d = json.loads(data)
#
#        # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
#        if d['rx_array_size'] == rx_array_size and d['tx_array_size'] == tx_array_size: # and d['initial_alphabet_opt'] == initial_alphabet_opt:
#            count = count + 1
#            print (d['channel_samples_filename'])
#
#            num_of_levels = d['num_of_levels']
#            initial_alphabet_opt = d['initial_alphabet_opt']
#            data_dict[num_of_levels] = {} 
#
#    print (data_dict)
#
#    for pathfile_id, pathfile in pathfiles.items():
#        with open(pathfile) as result:
#            data = result.read()
#            d = json.loads(data)
#    
#        # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
#        if d['rx_array_size'] == rx_array_size and d['tx_array_size'] == tx_array_size: # and d['initial_alphabet_opt'] == initial_alphabet_opt:
#            count = count + 1
#    
#            num_of_levels = d['num_of_levels']
#
#            distortion_by_round = d['mean_distortion_by_round'] 
#            mean_distortion = None
#            for r, mean_distortion in distortion_by_round.items():
#                mean_distortion = dict2matrix(mean_distortion)
#
#            initial_alphabet_opt = d['initial_alphabet_opt']
#            data_dict[num_of_levels][initial_alphabet_opt] = mean_distortion[-1]
#
#    data_keys_ordered = sorted(data_dict) #, key=lambda k: k['num_of_levels'])
#    data_new = {}
#    for k in data_keys_ordered:
#        data_new[k] = data_dict[k]
#    print (data_new)
#
#    color_dict = {'random': 'green', 'random_from_samples': 'red', 'xiaoxiao': 'orange', 'katsavounidis': 'blue' }
#    df = pd.DataFrame(data_new).T
#    sorted_cols = sorted(df.columns.tolist())
#    df = df[sorted_cols]
#    df.plot.bar(color = color_dict)
#    print (df)
#    print (title)
#    plt.title(title)
#    plt.legend(loc='best')
#    plt.show()
