import sys
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import seaborn as sns


if __name__ == '__main__':
    n = 8
    user_filter_set = {'filter1': {'ds_name': 's000', 'rx_array_size': n, 'tx_array_size': n, 'initial_alphabet_opt': 'random', 'num_of_levels': 4},
                       'filter2': {'ds_name': 's000', 'rx_array_size': n, 'tx_array_size': n, 'initial_alphabet_opt': 'random', 'num_of_levels': 8},
                       'filter3': {'ds_name': 's000', 'rx_array_size': n, 'tx_array_size': n, 'initial_alphabet_opt': 'random', 'num_of_levels': 16},
                       'filter4': {'ds_name': 's000', 'rx_array_size': n, 'tx_array_size': n, 'initial_alphabet_opt': 'random', 'num_of_levels': 32},
                       'filter5': {'ds_name': 's000', 'rx_array_size': n, 'tx_array_size': n, 'initial_alphabet_opt': 'random', 'num_of_levels': 64},
                       'filter6': {'ds_name': 's000', 'rx_array_size': n, 'tx_array_size': n, 'initial_alphabet_opt': 'random', 'num_of_levels': 128},
                       'filter7': {'ds_name': 's000', 'rx_array_size': n, 'tx_array_size': n, 'initial_alphabet_opt': 'random', 'num_of_levels': 256},
                       'filter8': {'ds_name': 's000', 'rx_array_size': n, 'tx_array_size': n, 'initial_alphabet_opt': 'random', 'num_of_levels': 512}}
    # We can have more than one filter:
    #user_filter_set = {'filter1': {'ds_name': 's004', 'rx_array_size': 4, 'tx_array_size': 4, 'initial_alphabet_opt': 'random', 'num_of_levels': 16}}
    #                   'filter2': {'ds_name': 's009', 'rx_array_size': 4, 'tx_array_size': 4, 'initial_alphabet_opt': 'xiaoxiao', 'num_of_levels': 4}}
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

                distortion_by_round = d['mean_distortion_by_round']
                for r, mean_distortion in distortion_by_round.items():
                    mean_distortion_dict[filter_id] = dict2matrix(mean_distortion)


    #mean_distortion = np.arange(mean_distortion)
    #print (type(mean_distortion))
    fig, ax = plt.subplots()
    for filter_id, mean_distortion in mean_distortion_dict.items():
        pass
        ax.plot(10 * np.log10(mean_distortion))
    plt.show()
#    profile_pathfile = 'profile-rt-s002.json' 
#    with open(profile_pathfile) as profile:
#        data = profile.read()
#        d = json.loads(data)
#
#    prefix_pathfiles = d['results_directory']
#    result_files = os.listdir(prefix_pathfiles)
#    pathfiles = check_files(prefix_pathfiles, result_files)
#    print ('# of json files: ', len(pathfiles))
#    #num_of_levels_opts = d['num_of_levels_opts']
#
#    #gridsize = (2, 3)
#    #fig = plt.figure(figsize=(8, 12))
#    
#    n = int(sys.argv[1])
#
#    l = int(sys.argv[2])
#    num_of_levels = l
#
#    rx_array_size = n
#    tx_array_size = n
#
#    initial_alphabet_opt = str(sys.argv[3])
#
#    count = 0
#
#    distortion_data = {}
#    for pathfile_id, pathfile in pathfiles.items():
#        with open(pathfile) as result:
#            data = result.read()
#            d = json.loads(data)
#
#        # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
#        if d['rx_array_size'] == rx_array_size and d['tx_array_size'] == tx_array_size and d['initial_alphabet_opt'] == initial_alphabet_opt and d['num_of_levels'] == num_of_levels:
#        #if d['rx_array_size'] == rx_array_size and d['tx_array_size'] == tx_array_size:
#            count = count + 1
#            num_of_levels = d['num_of_levels']
#            distortion_by_round = d['mean_distortion_by_round'] 
#
#            mean_distortion = None
#            for r, mean_distortion in distortion_by_round.items():
#                mean_distortion = dict2matrix(mean_distortion)
#
#            distortion_data[num_of_levels] = mean_distortion
#        #fig, ax = plt.subplots()
#        #ax.ticklabel_format(useOffset=False)
#        #ax.plot(mean_distortion)
#        #plt.ylabel('mse')
#        #plt.xlabel('# iterations')
#        #plt.title(f'initial alphabet opt: {initial_alphabet_opt}')
#        #plt.show()   
#
#    #print (f'distortion_data.keys(): \n{distortion_data.keys()}')
#    #print (f'distortion_data.values(): \n{distortion_data.values()}')
#    print (f'count: {count}')
#
#    linestyle_tuple = [
#                        (0, (1, 10)), 
#                        (0, (1, 1)), 
#                        (0, (5, 10)), 
#                        (0, (5, 5)), 
#                        (0, (5, 1)), 
#                        (0, (3, 10, 1, 10)), 
#                        (0, (3, 5, 1, 5)), 
#                        (0, (3, 1, 1, 1)), 
#                        (0, (3, 5, 1, 5, 1, 5)), 
#                        (0, (3, 10, 1, 10, 1, 10)), 
#                        (0, (3, 1, 1, 1, 1, 1))]
#
#
#
#    distortion_data = dict(sorted(distortion_data.items()))
#
#
#    count_ls = 0
#
#    
#    text = f''
#    for k, v in distortion_data.items():
#    #for vl in distortion_data:
#        print ('----')
#        count_ls += 1
#        #k = int(vl[0])
#        print (k)
#        #v = vl[1]
#        print (len(v))
#        plt.plot(v, linestyle = linestyle_tuple[count_ls], label=f'L={k}')
#        text = text +  f'\n({k}, {len(v)}, {v[-1]})'
#        #plt.text(len(v), v[len(v)-1], text, fontsize=8, color='purple')
#    plt.text(5, 0.009, text, fontsize=8, color='purple')
#    print (text)
#
#    plt.xlabel('Número de interações' )
#    plt.ylabel('Distorção (MSE)')
#    plt.title(f'N = {n}, inicialização: {initial_alphabet_opt}')
#    plt.legend()
#    plt.show()
#
#    #max_len = -np.Inf
#    #distortion_data_on_list = {}
#    #for k, v in distortion_data.items():
#    #    v = 10 * np.log(v)
#    #    print (type(v))
#    #    distortion_data_on_list[k] = v.tolist() # = [i for i in v]
#    #    print (f'k{k}: len(v) = {len(v)}')
#    #    if len(v) > max_len:
#    #        max_len = len(v)
#
#    #print (f'max_len: {max_len}')
#    #print (f'**********************')
#
#    #for k, v in distortion_data_on_list.items():
#    #    print (type(v))
#    #    print (f'k{k}: len(v) = {len(v)}')
#    #    #print (f'v: \n{v}')
#    #    while len(v) < max_len:
#    #        pass
#    #        v.append(v[-1])
#
#    #print (f'----------------------')
#
#    #for k, v in distortion_data_on_list.items():
#    #    print (type(v))
#    #    print (f'k{k}: len(v) = {len(v)}')
# 
#    #df = pd.DataFrame(distortion_data_on_list)       
#    #df.plot()
#    #plt.xlabel('# of iteration')
#    #plt.ylabel('min mse (db)')
#    #plt.title(f'{prefix_pathfiles}')
#    #plt.show()
##
##    x_labels = initial_alphabet_opts_available_min_distortion.keys() #["katsavounidis", "xiaoxiao", "sa", "unitary", "random from samples", "random"]
##    interval_list = initial_alphabet_opts_available_min_distortion.values()
##    data = initial_alphabet_opts_available_min_distortion
##    #data_list = [{'initial_alphabet_method': k, 'min_mse': v} for k, v in data.items()]
##    #sorted_data = sorted(data_list, key=lambda k: k['min_mse'])
##    #sorted_data = [i.keys() for i in sorted_data]
##    #print (f'sorted_data: \n{sorted_data}')
##    df = pd.DataFrame(data=data.values(), index=data.keys(), columns=['min mse']) 
##    print (f'df: \n{df}')
##     
##    df.plot.bar(rot=0) #sns.barplot(df)
##    plt.xlabel('initial alphabet opts')
##    plt.title(f'{prefix_pathfiles}')
##    plt.show()
##    #interval_list = [v for v in interval_list]
##    #interval_list = np.array(interval_list)
##    #interval_list = interval_list/norm(interval_list)
##    ##fig, ax = plt.subplots()
##    ##ax.ticklabel_format(useOffset=False)
##    ##plt.errorbar(x_labels, interval_list, fmt='o')
##    ##plt.show()
##
##if __name__ == '__main__':
##
##    #profile_pathfile = 'profile.json' 
##    #result_pathfile = sys.argv[1]
##
##    with open(result_pathfile) as result:
##        data = result.read()
##        d = json.loads(data)
##
##    initial_alphabet_opt = d['initial_alphabet_opt']
##    distortion_by_round = d['mean_distortion_by_round'] 
##
##    fig, ax = plt.subplots()
##    ax.ticklabel_format(useOffset=False)
##    for r, mean_distortion in distortion_by_round.items():
##        mean_distortion = dict2matrix(mean_distortion)
##        ax.plot(mean_distortion)
##    plt.ylabel('mse')
##    plt.xlabel('# iterations')
##    plt.title(f'initial alphabet opt: {initial_alphabet_opt}')
##    plt.show()
