#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@author: sergiossc@gmail.com
"""
import sys
#sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/vq')

#import concurrent.futures
import numpy as np
#import uuid
#import json
#import os
from utils import throw_lloyd

if __name__ == '__main__':
    
    channel_samples_filename = str(sys.argv[1])
    samples = np.load(channel_samples_filename)

    initial_alphabet_opt = str(sys.argv[2])
    num_of_levels = int(sys.argv[3])
    #phase_shift_resolution = int(sys.argv[4])
    distortion_measure_opt = str(sys.argv[4])
    max_num_of_interactions = int(sys.argv[5])
    results_dir = str(sys.argv[6])
    trial_id = str(sys.argv[7])
    trial_random_seed = int(sys.argv[8])


    #p = {'channel_samples': samples, 'channel_samples_filename': channel_samples_filename, 'initial_alphabet_opt':initial_alphabet_opt, 'num_of_levels':num_of_levels, 'phase_shift_resolution':phase_shift_resolution,'distortion_measure_opt':distortion_measure_opt, 'max_num_of_interactions':max_num_of_interactions, 'results_dir': results_dir, 'trial_id': trial_id, 'trial_random_seed': trial_random_seed}
    p = {'channel_samples': samples, 'channel_samples_filename': channel_samples_filename, 'initial_alphabet_opt':initial_alphabet_opt, 'num_of_levels':num_of_levels, 'distortion_measure_opt':distortion_measure_opt, 'max_num_of_interactions':max_num_of_interactions, 'results_dir': results_dir, 'trial_id': trial_id, 'trial_random_seed': trial_random_seed}


    #print (p)
    #result = run_lloyd_gla(p)
    result = throw_lloyd(p)
