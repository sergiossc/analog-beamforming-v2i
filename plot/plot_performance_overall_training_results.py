import json
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import seaborn as sns

if __name__ == '__main__':

    profile_pathfile = 'profile.json' 
    with open(profile_pathfile) as profile:
        data = profile.read()
        d = json.loads(data)

    prefix_pathfiles = d['results_directory']
    result_files = os.listdir(prefix_pathfiles)
    pathfiles = check_files(prefix_pathfiles, result_files)
    print ('# of json files: ', len(pathfiles))
    initial_alphabet_opts_available = d['initial_alphabet_opts_available']

    #gridsize = (2, 3)
    #fig = plt.figure(figsize=(8, 12))
    
    distortion_data = {}
    for pathfile_id, pathfile in pathfiles.items():
        with open(pathfile) as result:
            data = result.read()
            d = json.loads(data)

        # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
        initial_alphabet_opt = d['initial_alphabet_opt']
        distortion_by_round = d['mean_distortion_by_round'] 

        
        for r, mean_distortion in distortion_by_round.items():
            mean_distortion = dict2matrix(mean_distortion)

        distortion_data[initial_alphabet_opt] = mean_distortion
        #fig, ax = plt.subplots()
        #ax.ticklabel_format(useOffset=False)
        #ax.plot(mean_distortion)
        #plt.ylabel('mse')
        #plt.xlabel('# iterations')
        #plt.title(f'initial alphabet opt: {initial_alphabet_opt}')
        #plt.show()   

    #print (f'distortion_data.keys(): \n{distortion_data.keys()}')
    #print (f'distortion_data.values(): \n{distortion_data.values()}')

    max_len = -np.Inf
    distortion_data_on_list = {}
    for k, v in distortion_data.items():
        v = 10 * np.log(v)
        print (type(v))
        distortion_data_on_list[k] = v.tolist() # = [i for i in v]
        print (f'k{k}: len(v) = {len(v)}')
        if len(v) > max_len:
            max_len = len(v)

    print (f'max_len: {max_len}')
    print (f'**********************')

    for k, v in distortion_data_on_list.items():
        print (type(v))
        print (f'k{k}: len(v) = {len(v)}')
        #print (f'v: \n{v}')
        while len(v) < max_len:
            pass
            v.append(v[-1])

    print (f'----------------------')

    for k, v in distortion_data_on_list.items():
        print (type(v))
        print (f'k{k}: len(v) = {len(v)}')
 
    df = pd.DataFrame(distortion_data_on_list)       
    df.plot()
    plt.xlabel('# of iteration')
    plt.ylabel('min mse (db)')
    plt.title(f'{prefix_pathfiles}')
    plt.show()
#
#    x_labels = initial_alphabet_opts_available_min_distortion.keys() #["katsavounidis", "xiaoxiao", "sa", "unitary", "random from samples", "random"]
#    interval_list = initial_alphabet_opts_available_min_distortion.values()
#    data = initial_alphabet_opts_available_min_distortion
#    #data_list = [{'initial_alphabet_method': k, 'min_mse': v} for k, v in data.items()]
#    #sorted_data = sorted(data_list, key=lambda k: k['min_mse'])
#    #sorted_data = [i.keys() for i in sorted_data]
#    #print (f'sorted_data: \n{sorted_data}')
#    df = pd.DataFrame(data=data.values(), index=data.keys(), columns=['min mse']) 
#    print (f'df: \n{df}')
#     
#    df.plot.bar(rot=0) #sns.barplot(df)
#    plt.xlabel('initial alphabet opts')
#    plt.title(f'{prefix_pathfiles}')
#    plt.show()
#    #interval_list = [v for v in interval_list]
#    #interval_list = np.array(interval_list)
#    #interval_list = interval_list/norm(interval_list)
#    ##fig, ax = plt.subplots()
#    ##ax.ticklabel_format(useOffset=False)
#    ##plt.errorbar(x_labels, interval_list, fmt='o')
#    ##plt.show()
#
#if __name__ == '__main__':
#
#    #profile_pathfile = 'profile.json' 
#    #result_pathfile = sys.argv[1]
#
#    with open(result_pathfile) as result:
#        data = result.read()
#        d = json.loads(data)
#
#    initial_alphabet_opt = d['initial_alphabet_opt']
#    distortion_by_round = d['mean_distortion_by_round'] 
#
#    fig, ax = plt.subplots()
#    ax.ticklabel_format(useOffset=False)
#    for r, mean_distortion in distortion_by_round.items():
#        mean_distortion = dict2matrix(mean_distortion)
#        ax.plot(mean_distortion)
#    plt.ylabel('mse')
#    plt.xlabel('# iterations')
#    plt.title(f'initial alphabet opt: {initial_alphabet_opt}')
#    plt.show()
