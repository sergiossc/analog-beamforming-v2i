import sys
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import seaborn as sns
import uuid

plt.rcParams.update({'font.size':9})

if __name__ == '__main__':
    n_set = [4, 8, 16, 32, 64]
    #n_set = [4]
    setup_dict = {4:'i', 8:'ii', 16:'iii', 32:'iv', 64:'v'}
    small_axis_y_lim_dict = {4:-45, 8:-40, 16:-35, 32:-35, 64:-33}

    initial_alphabet_set = ['xiaoxiao', 'random_from_samples', 'random']
    #initial_alphabet_set = ['katsavounidis', 'xiaoxiao', 'random_from_samples', 'random']
    #initial_alphabet_set_label_pt_dict = {'katsavounidis': 'katsavounidis', 'xiaoxiao':'xiaoxiao', 'random_from_samples':'aleatório(amostras)', 'random':'aleatório'}
    initial_alphabet_set_label_pt_dict = {'xiaoxiao':'xiaoxiao', 'random_from_samples':'aleatório(amostras)', 'random':'aleatório'}
    num_levels_set = [4, 8, 16, 32, 64, 128, 256, 512]
    dataset_name = 's002'

    for n in n_set:
        user_filter_set = {}
        for num_levels in num_levels_set:
            for initial_alphabet in initial_alphabet_set:
                user_filter_id = uuid.uuid4()
                user_filter = {'ds_name': dataset_name, 'rx_array_size': n, 'tx_array_size': n, 'initial_alphabet_opt': initial_alphabet, 'num_of_levels': num_levels}
                user_filter_set[user_filter_id] = user_filter

        min_mean_distortion_dict = {}
        for n_levels in num_levels_set:
            min_mean_distortion_dict[n_levels] = {}
        #    min_mean_distortion_dict[filter_id] = None
    
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
                    mean_distortion = None
                    for r, v in distortion_by_round.items():
                        mean_distortion = dict2matrix(v)
                    min_mean_distortion_dict[num_of_levels][initial_alphabet] = mean_distortion[-1] 
                    print (mean_distortion[-1])
    
        pass
        data_keys_ordered = sorted(min_mean_distortion_dict) #, key=lambda k: k['num_of_levels'])
        data_new = {}
        for k in data_keys_ordered:
             print (f'k------------------> {k}')
             data_new[k] = min_mean_distortion_dict[k]
        print (data_new)

        color_dict = {'random': 'green', 'random_from_samples': 'red', 'xiaoxiao': 'orange', 'katsavounidis': 'blue' }
        #color_dict = {'random': 'green', 'random_from_samples': 'red'}
        df = pd.DataFrame(data_new).T
        #sorted_cols = sorted(df.columns.tolist())
        #sorted_cols = ['katsavounidis', 'random', 'random_from_samples', 'xiaoxiao']
        sorted_cols = ['xiaoxiao', 'random_from_samples', 'random']
        #print (f'sorted_cols: {sorted_cols}')
        sorted_cols_pt = [initial_alphabet_set_label_pt_dict[v] for v in sorted_cols]
        df = df[sorted_cols]
        df.plot.bar(color = color_dict, rot=0)
        #print (df)
        #print (df.columns)
        #print (title)
        #plt.title(f'Experimento {setup_dict[n]} (N={n})')
        plt.xlabel(f'Número de codewords (L)', fontsize='9')
        plt.ylabel(r'Valor mínimo da distorção média $d_m$', fontsize='9') 
        plt.legend(sorted_cols_pt, loc='best', fontsize='9')
        plt.grid(True)
        #plt.legend(loc='best', fontsize='11')
        #image_filename = f'training-{dataset_name}-distortion-iteractions-setup_{setup_dict[n]}.png'
        image_filename = f'training-{dataset_name}-distortion-by-num-of-cw-n{n}_new.png'
        print (image_filename)
        plt.savefig(image_filename, bbox_inches='tight')
        #plt.show()                   

