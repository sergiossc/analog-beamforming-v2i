import uuid
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from utils import *

#def check_files(prefix, episodefiles):
#    pathfiles = {}
#    for ep_file in episodefiles:
#        pathfile = prefix + str('/') + str(ep_file)
#        ep_file_status = False
#        try:
#            current_file = open(pathfile)
#            ep_file_status = True
#            #print("Sucess.")
#        except IOError:
#            print("File not accessible: ", pathfile)
#        finally:
#            current_file.close()
#
#        if ep_file_status:
#            ep_file_id = uuid.uuid4()
#            pathfiles[ep_file_id] = pathfile
# 
#    return pathfiles
#
#
#def decode_mean_distortion(mean_distortion_dict):
#    mean_distortion_list = []
#    for iteration, mean_distortion in mean_distortion_dict.items():
#        mean_distortion_list.append(mean_distortion)
#    return mean_distortion_list


if __name__ == '__main__':

    profile_pathfile = 'profile.json' 

    with open(profile_pathfile) as profile:
        data = profile.read()
        d = json.loads(data)

    prefix_pathfiles = d['results_directory']
    result_files = os.listdir(prefix_pathfiles)
    pathfiles = check_files(prefix_pathfiles, result_files)
    print ('# of json files: ', len(pathfiles))
    # From here it is going to open each json file to see each parameters and data from algorithm perform. May you should to implement some decode or transate functions to deal with json data from files to python data format. There are some decode functions on utils library. 
    #trial_result = (initial_alphabet_opt, distortion_measure_opt, num_of_levels, variance_of_samples, norm)
    list_of_elements = [4, 8, 16, 32, 64]

    for n in list_of_elements:

        occurences = []
        samples_random_seeds = {}
    
        katsavounidis_results = []
        xiaoxiao_results = []
        unitary_until_num_of_elements_results = []
        random_from_samples_results = []
        sa_results = []
        random_results = []
    
        for pathfile_id, pathfile in pathfiles.items():
            with open(pathfile) as result:
                data = result.read()
                d = json.loads(data)
    
            # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
            initial_alphabet_opt = d['initial_alphabet_opt']
            variance_of_samples = d['variance_of_samples']
            distortion_measure_opt = d['distortion_measure_opt']
            initial_alphabet_opt = d['initial_alphabet_opt']
            num_of_elements = d['num_of_elements']
            num_of_levels = num_of_elements 
            num_of_samples = d['num_of_samples']
            samples_random_seed = d['samples_random_seed']
            mean_distortion_by_round = d['mean_distortion_by_round']
    
            #normal_vector = np.ones(num_of_levels) * (num_of_samples/num_of_levels)
            #sets = d['sets']
            #set_vector = []
            #for k, v in sets.items():
            #    set_vector.append(v)
            #set_vector = np.array(set_vector)
       
            #norm =  np.sqrt(np.sum(np.power(np.abs(set_vector - normal_vector), 2)))
            #if norm == 0 and num_of_elements == 9 and variance_of_samples == 1.0 and initial_alphabet_method == 'katsavounidis': 
            #if  norm == 0 and num_of_elements == 4 and variance_of_samples == 0.1 and initial_alphabet_method == 'katsavounidis'
            #if  variance_of_samples == 0.1 and num_of_elements == 4 and initial_alphabet_opt == 'katsavounidis':
                #trial_info = {'norm': norm}
                #occurences.append(trial_info)
            #if  num_of_elements == 4:
            samples_random_seeds[int(samples_random_seed)] = 1
    
            if initial_alphabet_opt == 'katsavounidis' and num_of_elements == n:
                katsavounidis_results.append(pathfile)
    
            if initial_alphabet_opt == 'xiaoxiao' and num_of_elements == n:
                xiaoxiao_results.append(pathfile)
    
            if initial_alphabet_opt == 'sa' and num_of_elements == n:
                sa_results.append(pathfile)
    
            if initial_alphabet_opt == 'unitary_until_num_of_elements' and num_of_elements == n:
                unitary_until_num_of_elements_results.append(pathfile)
    
            if initial_alphabet_opt == 'random_from_samples' and num_of_elements == n:
                random_from_samples_results.append(pathfile)
    
            if initial_alphabet_opt == 'random' and num_of_elements == n:
                random_results.append(pathfile)
    
            occurences.append(1)
    
        print(f'******************************************')
        print(f'n-value: {n}')
        print(f'All occurences length: {len(occurences)}')
        print(f'katsavounidis_results length: {len(katsavounidis_results)}')
        print(f'xiaoxiao_results length: {len(xiaoxiao_results)}')
        print(f'sa_results length: {len(sa_results)}')
        print(f'unitary_until_num_of_elements_results length: {len(unitary_until_num_of_elements_results)}')
        print(f'random_from_samples_results length: {len(random_from_samples_results)}')
        print(f'random_results length: {len(random_results)}')
