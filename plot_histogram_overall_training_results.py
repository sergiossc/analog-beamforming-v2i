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
    num_trials = len(pathfiles)
    initial_alphabet_opts_available = d['initial_alphabet_opts_available']

    #gridsize = (2, 3)
    #fig = plt.figure(figsize=(8, 12))

    index = []
    sets_mtx = []

    for pathfile_id, pathfile in pathfiles.items():
        with open(pathfile) as result:
            data = result.read()
            d = json.loads(data)

        # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
        initial_alphabet_opt = d['initial_alphabet_opt']
        print (initial_alphabet_opt)
        index.append(initial_alphabet_opt)

        sets_dict = d['sets'] 
        print (f'sets: {sets_dict}')

        sets_arr = dict2matrix(sets_dict)
        sets_mtx.append(sets_arr.tolist())

    sets_mtx = np.matrix(sets_mtx)
    sets_mtx = sets_mtx.T
    n_row, n_col = np.shape(sets_mtx)
    print (f'nrow. ncol: {n_row, n_col}')
    print (f'sets: {sets_mtx}')

    index = [i for i in index]
    print (index)

    data = {}
    for i in range(n_row):
        k = f'cw' + str(i)
        data[k] = sets_mtx[i].tolist()[0]
    print (f'data: \n{data}')
    df = pd.DataFrame(data, index=index)
    print (f'df: \n{df}')
    df.plot.bar(rot=0)
    plt.title(f'{prefix_pathfiles}')
    plt.xlabel(f'initial alphabet opts')
    plt.ylabel(f'# of samples')
    plt.show()
