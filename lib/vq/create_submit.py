#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sergiossc@gmail.com
"""
import numpy as np
import uuid
import json
import os
from utils import *
import sys

if __name__ == '__main__':


    profile_pathfile = 'profile.json' 

    with open(profile_pathfile) as profile:
        data = profile.read()
        d = json.loads(data)

    # Read information from 'profile.json' file
    num_of_elements = d['number_of_elements']
    variance_of_samples_values = d['variance_of_samples_values']
    initial_alphabet_opts = d['initial_alphabet_opts']
    distortion_measure_opts = d['distortion_measure_opts']
    num_of_trials = d['num_of_trials']
    num_of_samples = d['num_of_samples']
    max_num_of_interactions = d['max_num_of_interactions']
    results_dir = d['results_directory'] 
    use_same_samples_for_all = d['use_same_samples_for_all']
    percentage_of_sub_samples = d['percentage_of_sub_samples']

    parms = []
    for n_elements in num_of_elements:
        for variance in variance_of_samples_values:
            for initial_alphabet_opt in initial_alphabet_opts:
                for distortion_measure_opt in distortion_measure_opts:

                    if use_same_samples_for_all:
                        np.random.seed(789)
                        random_seeds = np.random.choice(100000, num_of_trials, replace=False)
                        np.random.seed(None)
                    else:
                        random_seeds = np.random.choice(100000, num_of_trials, replace=False)
                        
                    for n in range(num_of_trials):
                        p = {'num_of_elements': n_elements, 'variance_of_samples': variance, 'initial_alphabet_opt':initial_alphabet_opt, 'distortion_measure_opt':distortion_measure_opt, 'num_of_samples':num_of_samples, 'max_num_of_interactions':max_num_of_interactions, 'results_dir': results_dir, 'instance_id': str(uuid.uuid4()), 'percentage_of_sub_samples': percentage_of_sub_samples, 'samples_random_seed': random_seeds[n], 'trial_random_seed': np.random.choice(10000, 1)[0]}
                        parms.append(p)
  
        
    args = ''
    args = '\nexecutable = ' + str('run_lloyd.py') 
    args += '\nlog = ' + str('run_lloyd.$(Cluster).$(Process).out')
    args += '\nerror = ' + str('run_lloyd.$(Cluster).$(Process).err')
    #args += '\noutput = ' +
    args += '\nshould_transfer_files = ' + str('Yes')
    args += '\ntransfer_input_files = ' + str('utils.py')
    args += '\nwhen_to_transfer_output = ' + str('ON_EXIT')

    for p in parms:
        print ('***************')

        num_of_elements = p['num_of_elements']
        variance_of_samples = p['variance_of_samples']
        initial_alphabet_opt = p['initial_alphabet_opt']
        distortion_measure_opt = p['distortion_measure_opt']
        num_of_samples = p['num_of_samples']
        max_num_of_interactions = p['max_num_of_interactions']
        results_dir = p['results_dir']
        instance_id = p['instance_id']
        percentage_of_sub_samples = p['percentage_of_sub_samples']
        samples_random_seed = p['samples_random_seed']
        trial_random_seed = p['trial_random_seed']
    
    
        #args = 'run_lloyd_gla(' + str(num_of_elements) + ', ' +  str(variance_of_samples) + ', \'' + str(initial_alphabet_opt) + '\', \'' + str(distortion_measure_opt) + '\', ' + str(num_of_samples) + ', ' + str(max_num_of_interactions) + ', \'' + str(results_dir) + '\', \'' + str(instance_id) + '\', ' + str(percentage_of_sub_samples) + ', ' + str(samples_random_seed) + ', ' + str(trial_random_seed) + ')'
        args += '\narguments = ' + str(num_of_elements) + ' ' +  str(variance_of_samples) + ' ' + str(initial_alphabet_opt) + ' ' + str(distortion_measure_opt) + ' ' + str(num_of_samples) + ' ' + str(max_num_of_interactions) + ' ' + str(results_dir) + ' ' + str(instance_id) + ' ' + str(percentage_of_sub_samples) + ' ' + str(samples_random_seed) + ' ' + str(trial_random_seed) + ' \nqueue'



    f = open('auto-submit.sh', 'w+')
    f.write(args)
    f.close()
    print (args)






    
    #print ('# of cpus: ', os.cpu_count())
    #print ('# of parms: ', len(parms))
    
    #with concurrent.futures.ProcessPoolExecutor() as e:
    #    for p, r in zip(parms, e.map(run_lloyd_gla, parms)):
    #        print ('parm ' + str(p['instance_id']) + ' returned  ' + str(r))
