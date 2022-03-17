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
    min_distorsion = np.Inf
    min_distorsion_pathfile = ''
    initial_alphabet_opts_available = d['initial_alphabet_opts_available']
    print (f'initial_alphabet_opts_available: {initial_alphabet_opts_available}')
    initial_alphabet_opts_available_min_distortion = {k: np.Inf for k in initial_alphabet_opts_available}
    print (f'initial_alphabet_opts_available_min_distortion: {initial_alphabet_opts_available_min_distortion}')


    for pathfile_id, pathfile in pathfiles.items():
        with open(pathfile) as result:
            data = result.read()
            d = json.loads(data)

        # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
        initial_alphabet_opt = d['initial_alphabet_opt']
        mean_distortion_by_round = d['mean_distortion_by_round']
    
        
        #getting min distortion result
        for k1, v1 in mean_distortion_by_round.items():
            for k, v in v1.items():
                if v < initial_alphabet_opts_available_min_distortion[initial_alphabet_opt]:
                    initial_alphabet_opts_available_min_distortion[initial_alphabet_opt] = v
                if v < min_distorsion:
                    min_distorsion = v
                    min_distorsion_pathfile = pathfile


    print (f'*****************************************')
    print (f'initial_alphabet_opts_available_min_distortion: {initial_alphabet_opts_available_min_distortion}')
    print (min_distorsion_pathfile)

    x_labels = initial_alphabet_opts_available_min_distortion.keys() #["katsavounidis", "xiaoxiao", "sa", "unitary", "random from samples", "random"]
    interval_list = initial_alphabet_opts_available_min_distortion.values()
    data = initial_alphabet_opts_available_min_distortion
    #data_list = [{'initial_alphabet_method': k, 'min_mse': v} for k, v in data.items()]
    #sorted_data = sorted(data_list, key=lambda k: k['min_mse'])
    #sorted_data = [i.keys() for i in sorted_data]
    #print (f'sorted_data: \n{sorted_data}')
    df = pd.DataFrame(data=data.values(), index=data.keys(), columns=['min mse']) 
    print (f'df: \n{df}')
     
    df.plot.bar(rot=0) #sns.barplot(df)
    plt.xlabel('initial alphabet opts')
    plt.title(f'{prefix_pathfiles}')
    plt.show()
    #interval_list = [v for v in interval_list]
    #interval_list = np.array(interval_list)
    #interval_list = interval_list/norm(interval_list)
    ##fig, ax = plt.subplots()
    ##ax.ticklabel_format(useOffset=False)
    ##plt.errorbar(x_labels, interval_list, fmt='o')
    ##plt.show()
