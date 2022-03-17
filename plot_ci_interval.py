import os
import sys
import json
import uuid
from utils import compare_filter, get_all_result_json_pathfiles, get_all_possible_filters, conf_interval
import matplotlib.pyplot as plt

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

    


    x_set = []
    labels_set = []
    ci_values_set = []
    x_info_list = []
    
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
                x_set.append(d[k1])
                labels_set.append(v1)
                x_set_mean, x_set_ci_values = conf_interval(d[k1], conf_level)
                x_info = {'x': d[k1], 'label': f'{v1}', 'x_mean': x_set_mean, 'ci': x_set_ci_values}
                x_info_list.append(x_info)
                ci_values_set.append(x_set_ci_values)
    
        keys_list2 = {'katsavounidis':'katsavounidis', 'xiaoxiao':'xiaoxiao', 'random_from_samples':'aleatório(amostras)', 'random': 'aleatório'}
        #keys_list3 = {'bf_gain_list':'', 'bf_gain_egt_list':'(EGT)'}
        keys_list3 = {'bf_gain_egt_list':'(EGT)'}
        for k2, v2 in keys_list2.items():
            for k3, v3 in keys_list3.items():
                x_set.append(d[k2][k3])
                labels_set.append(f'{v2} {v3}, L={num_levels}')
                x_set_mean, x_set_ci_values = conf_interval(d[k2][k3], conf_level)
                x_info = {'x': d[k2][k3], 'label': f'{v2} {v3}, L={num_levels}', 'x_mean': x_set_mean, 'ci': x_set_ci_values}
                x_info_list.append(x_info)
                ci_values_set.append(x_set_ci_values)

        d_keys = list(d.keys())
        #print (d_keys)
        for d_k in d_keys:
            if d_k == 'dft':
                pass
                dft_num_levels = d['dft']['num_of_levels']
                dft_x = d['dft']['bf_gain_list']
                x_set.append(dft_x)
                labels_set.append(f'DFT, L={dft_num_levels}')
                x_set_mean, x_set_ci_values = conf_interval(dft_x, conf_level)
                x_info = {'x': dft_x, 'label': f'DFT, L={dft_num_levels}', 'x_mean': x_set_mean, 'ci': x_set_ci_values}
                x_info_list.append(x_info)
                ci_values_set.append(x_set_ci_values)
                #x_info = {'x': d[d_k], 'label': f'DFT, L={num_levels}'}
                #x_info_list.append(x_info)


    
    fig1, ax1 = plt.subplots(figsize=(20,10))
    box = ax1.boxplot(x_set, conf_intervals=ci_values_set, notch=True, labels=labels_set)
    plt.xticks(rotation=90)
    #plt.show()
    print (fig_filename)
    plt.savefig(fig_filename, bbox_inches='tight')





#def plot_ci_boxplot_1(test_result_pathfile):
#
#    with open(test_result_pathfile) as result:
#        data = result.read()
#        d = json.loads(data)
#    pass
#    x_set = []
#    labels_set = []
#    keys_list1 = {'bf_gain_max_list':'BF gain', 'bf_gain_max_egt_list':'BF gain (EGT)'}
#    for k1,v1 in keys_list1.items():
#        x_set.append(d[k1])
#        labels_set.append(v1)
#
#    keys_list2 = {'katsavounidis':'katsavounidis', 'xiaoxiao':'xiaoxiao', 'random_from_samples':'aleatório(amostras)', 'random': 'aleatório'}
#    keys_list3 = {'bf_gain_list':'', 'bf_gain_egt_list':'(EGT)'}
#    for k2, v2 in keys_list2.items():
#        for k3, v3 in keys_list3.items():
#            x_set.append(d[k2][k3])
#            labels_set.append(f'{v2} {v3}')
#
#    fig1, ax1 = plt.subplots()
#    ax1.boxplot(x_set, labels=labels_set)
#    plt.xticks(rotation=90)
#    plt.show()
#


if __name__ == "__main__":

    #test_result_pathfile = str(sys.argv[1])
    datasets = ['s000', 's002', 's004', 's006', 's007', 's008', 's009']
    n_set = [4, 8, 16, 32, 64]
    #if True:
    for dataset_name in datasets:
        for n in n_set:
            my_filters = {'ds_name':dataset_name , 'rx_array_size':n, 'tx_array_size':n}
            #rootdir = "test_results.OLD"
            rootdir = "test_results"
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
            print (f'# of matched filters on /{rootdir}: {counter}')
            print (len(matched_results_dict))
            fig_filename = f'boxplot-{dataset_name}-n{n}.png'
            plot_ci_boxplot(matched_results_dict, fig_filename)
    #else:
    #   pass
    #   plot_ci_boxplot_1(test_result_pathfile)
 
#        print (d.keys())
#        bf_gain_ideal = d['bf_gain_max_list']
#        bf_gain_ideal_egt = d['bf_gain_max_egt_list']
#        bf_gain_ktsa = d['katsavounidis']['bf_gain_list']
#        bf_gain_ktsa_egt = d['katsavounidis']['bf_gain_egt_list']
#        #bf_gain_xiaoxiao = d['xiaoxiao']['bf_gain_list']
#        x_set = [bf_gain_ideal, bf_gain_ideal_egt]
#        conf_level = 0.90
#        sample_mean_gain_ideal, ci_values_gain_ideal = conf_interval(bf_gain_ideal, conf_level)
#        sample_mean_gain_egt_ideal, ci_values_gain_egt_ideal = conf_interval(bf_gain_ideal_egt, conf_level)
#        conf_intervals = [ci_values_gain_ideal, ci_values_gain_egt_ideal]
#        labels = ('BF ideal', 'BF ideal (EGT)')
#        #print (sample_mean)
#        fig1, ax1 = plt.subplots()
#        ax1.boxplot(x_set, conf_intervals=conf_intervals, notch=True, labels=labels)
#        plt.xticks(rotation=90)
#        plt.show()
