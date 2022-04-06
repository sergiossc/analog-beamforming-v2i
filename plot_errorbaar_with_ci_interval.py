import os
import sys
import json
import uuid
from utils import compare_filter, get_all_result_json_pathfiles, get_all_possible_filters, conf_interval
import matplotlib.pyplot as plt
import numpy as np

def plot_ci_boxplot(matched_results_dict, fig_filename):

    conf_level = 0.90

    num_levels_set = {4:[], 8:[], 16:[], 32:[], 64:[], 128:[], 256:[], 512:[]}
    for k, test_result_pathfile in matched_results_dict.items():

        with open(test_result_pathfile) as result:
            data = result.read()
            d = json.loads(data)
        pass
        num_levels = int(d['num_levels'])
        num_levels_set[num_levels].append(k)

    keys_seq = []
    for k, v in num_levels_set.items():
        #print (k)
        for i in v:
            #print (i)
            keys_seq.append(i)

    
    color_dict = {'dft': 'darkviolet', 'opt': 'black', 'random': 'green', 'random_from_samples': 'red', 'xiaoxiao': 'orange', 'katsavounidis': 'blue' }

    x_mean_list = []
    yerr_list = [] # (ci_values[1] - ci_values[0])/2
    labels_set = []
    color_list = []
    
    lock_max_gain = True

    for k in keys_seq:
        test_result_pathfile = matched_results_dict[k]

        with open(test_result_pathfile) as result:
            data = result.read()
            d = json.loads(data)
        pass
        num_levels = d['num_levels']

        if lock_max_gain:
            lock_max_gain = False
            #keys_list1 = {'bf_gain_max_list':'BF gain', 'bf_gain_max_egt_list':'BF gain (EGT)'}
            keys_list1 = {'bf_gain_max_egt_list':'BF gain (EGT)'}
            for k1,v1 in keys_list1.items():

                x_mean, x_ci_values = conf_interval(d[k1], conf_level)
                x_mean_list.append(x_mean)
                yerr = (x_ci_values[1] - x_ci_values[0])/2
                yerr_list.append(yerr)
                labels_set.append(v1)
                color_list.append(color_dict['opt'])
    
        #keys_list2 = {'katsavounidis':'katsavounidis', 'xiaoxiao':'xiaoxiao', 'random_from_samples':'aleatório(amostras)', 'random': 'aleatório'}
        keys_list2 = {'xiaoxiao':'xiaoxiao', 'random_from_samples':'aleatório(amostras)', 'random': 'aleatório'}
        #keys_list3 = {'bf_gain_list':'', 'bf_gain_egt_list':'(EGT)'}
        keys_list3 = {'bf_gain_egt_list':'(EGT)'}
        for k2, v2 in keys_list2.items():
            for k3, v3 in keys_list3.items():

                x_mean, x_ci_values = conf_interval(d[k2][k3], conf_level)
                x_mean_list.append(x_mean)
                yerr = (x_ci_values[1] - x_ci_values[0])/2
                yerr_list.append(yerr)
                labels_set.append(f'{v2} {v3}, L={num_levels}')
                color_list.append(color_dict[k2])
    



                #x = d[k2][k3]
                #labels_set.append(f'{v2} {v3}, L={num_levels}')

                #x_set.append(d[k2][k3])
                #x_set_mean, x_set_ci_values = conf_interval(d[k2][k3], conf_level)
                #x_info = {'x': d[k2][k3], 'label': f'{v2} {v3}, L={num_levels}', 'x_mean': x_set_mean, 'ci': x_set_ci_values}
                #x_info_list.append(x_info)
                #ci_values_set.append(x_set_ci_values)

        d_keys = list(d.keys())
        #print (d_keys)
        for d_k in d_keys:
            if d_k == 'dft':
                pass

                dft_num_levels = d['dft']['num_of_levels']
                dft_x = d['dft']['bf_gain_list']


                x_mean, x_ci_values = conf_interval(dft_x, conf_level)
                x_mean_list.append(x_mean)
                yerr = (x_ci_values[1] - x_ci_values[0])/2
                yerr_list.append(yerr)
                labels_set.append(f'DFT, L={dft_num_levels}')
                color_list.append(color_dict['dft'])
    





                #x = dft_x
                #labels_set.append(f'DFT, L={dft_num_levels}')


                #x_set.append(dft_x)
                #x_set_mean, x_set_ci_values = conf_interval(dft_x, conf_level)
                #x_info = {'x': dft_x, 'label': f'DFT, L={dft_num_levels}', 'x_mean': x_set_mean, 'ci': x_set_ci_values}
                #x_info_list.append(x_info)
                #ci_values_set.append(x_set_ci_values)
                #x_info = {'x': d[d_k], 'label': f'DFT, L={num_levels}'}
                #x_info_list.append(x_info)


    pass 
    #print (f'len(x_mean_list): {len(x_mean_list)}')
    #print (f'len(yerr_list): {len(yerr_list)}')
    fig1, ax1 = plt.subplots(figsize=(20,10))
    for i in range(len(x_mean_list)):
        ax1.errorbar(i, x_mean_list[i], yerr=yerr_list[i], capsize=5, marker='o', color=color_list[i])
    #fig1, ax1 = plt.subplots(figsize=(20,10))
    #ax1.errorbar(len(x_mean_list), x_mean_list, yerr=yerr_list, capsize=5)
    #box = ax1.boxplot(x_set, conf_intervals=ci_values_set, notch=True, labels=labels_set)
    plt.xticks(np.arange(len(labels_set)), labels_set)
    plt.xticks(rotation=90)
    plt.ylabel('Ganho de Beamforming (G)')
    plt.grid()
    #plt.show()
    print (fig_filename)
    plt.savefig(fig_filename, bbox_inches='tight')
    plt.close('all')



if __name__ == "__main__":

    #datasets = ['s000', 's002', 's004', 's006', 's007', 's008', 's009']
    datasets = ['s002']
    n_set = [4, 8, 16, 32, 64]
    #n_set = [64]
    #if True:
    for dataset_name in datasets:
        for n in n_set:
            print (f'dataset_name, n: {(dataset_name, n)}')
            my_filters = {'ds_name':dataset_name , 'rx_array_size':n, 'tx_array_size':n}
            #rootdir = "test_results"
            rootdir = "test_results_all_test_samples"
            #rootdir = "test_results"
            result_pathfiles_dict = get_all_result_json_pathfiles(rootdir)
            matched_results_dict = {}
        
            count = 0
        
            print (f'# of JSON result files: {len(result_pathfiles_dict)}')
            counter = 0
            bf_gain = []
            for k, pathfile in result_pathfiles_dict.items():
                
                with open(pathfile) as result:
                    data = result.read()
                    d = json.loads(data)
                pass
        
                ds_name = d['dataset_name']
                rx_array_size = d['n']
                tx_array_size = d['n']
                num_of_levels = d['num_levels']
        
                result_filter = {'ds_name':ds_name , 'rx_array_size':rx_array_size, 'tx_array_size':tx_array_size, 'num_of_levels':num_of_levels }
                
                if compare_filter(my_filters, result_filter): 
                    pass 
                    counter += 1
                    #print (result_filter)
                    matched_results_dict[k] = pathfile
                    #bf_gain = d['bf_gain_max_list']
            #print (f'# of matched filters on /{rootdir}: {counter}')
            #print (len(matched_results_dict))
            fig_filename = f'errorbar_plot-{dataset_name}-n{n}_new.png'
            plot_ci_boxplot(matched_results_dict, fig_filename)
    #else:
