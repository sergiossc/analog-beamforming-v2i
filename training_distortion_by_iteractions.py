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
    #n_set = [4, 8, 16, 32, 64]
    n_set = [4]
    setup_dict = {4:'i', 8:'ii', 16:'iii', 32:'iv', 64:'v'}
    small_axis_y_lim_dict = {4:-60, 8:-55, 16:-55, 32:-42, 64:-40}

    #initial_alphabet_set = ['katsavounidis', 'xiaoxiao', 'random_from_samples', 'random']
    initial_alphabet_set = ['xiaoxiao', 'random_from_samples', 'random']
    initial_alphabet_set_label_pt_dict = {'katsavounidis': 'katsavounidis', 'xiaoxiao':'xiaoxiao', 'random_from_samples':'aleatório(amostras)', 'random':'aleatório'}

    num_levels_set = [4] #, 8, 16, 32, 64, 128, 256, 512]
    #num_levels_set = [4, 8, 16, 32, 64, 128, 256, 512]

    dataset_name = 's002'
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
        fig, ax = plt.subplots(figsize=(20,10)) # (x_size, y_size)
        #plt.yscale("log")
        for filter_id, mean_distortion in mean_distortion_dict.items():
            pass
            label = initial_alphabet_set_label_pt_dict[user_filter_set[filter_id]['initial_alphabet_opt']]
            ax.plot(10 * np.log10(mean_distortion), marker=marker_dict[user_filter_set[filter_id]['num_of_levels']],  color=color_dict[user_filter_set[filter_id]['initial_alphabet_opt']], label=f"{label}, L={user_filter_set[filter_id]['num_of_levels']}")
            #ax.plot(mean_distortion, marker=marker_dict[user_filter_set[filter_id]['num_of_levels']],  color=color_dict[user_filter_set[filter_id]['initial_alphabet_opt']], label=f"{label}, L={user_filter_set[filter_id]['num_of_levels']}")
        plt.legend(loc='best', fontsize='11')
        plt.xlabel(f'Número de iterações', fontsize='11')
        plt.ylabel(r'Distorção média $d_m$ em db', fontsize='11') 
        #plt.title(f'Experimento {setup_dict[n]} (N={n})')

        ax_small = fig.add_axes([0.45, 0.5, 0.2, 0.35])
        for filter_id, mean_distortion in mean_distortion_dict.items():
            pass
            ax_small.plot(10 * np.log10(mean_distortion), marker=marker_dict[user_filter_set[filter_id]['num_of_levels']],  color=color_dict[user_filter_set[filter_id]['initial_alphabet_opt']])
            #ax_small.plot(mean_distortion, marker=marker_dict[user_filter_set[filter_id]['num_of_levels']],  color=color_dict[user_filter_set[filter_id]['initial_alphabet_opt']])
        ax_small.set_xlim(-1,15)
        ax_small.set_ylim(small_axis_y_lim_dict[n],2)
        image_filename = f'training-{dataset_name}-distortion-iteractions-setup_{setup_dict[n]}.png'
        print (image_filename)
        #plt.savefig(image_filename, bbox_inches='tight')
        plt.show()



