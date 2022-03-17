import os
import numpy as np
import sys
from scipy.constants import c
from numpy.linalg import svd, matrix_rank, norm
#from lib.vq.utils import norm
import matplotlib.pyplot as plt
from utils import richscatteringchnmtx, wf, siso_c, mimo_csir_c, mimo_csit_eigenbeamforming_c, mimo_csit_beamforming, get_orthonormal_basis
import json
from utils import decode_codebook, get_codebook, beamsweeping, check_files, beamsweeping2



if __name__ == "__main__":
   

    profile_pathfile = 'profile-rt.json' 
    with open(profile_pathfile) as profile:
        data = profile.read()
        d = json.loads(data)

    prefix_pathfiles = d['results_directory']
    result_files = os.listdir(prefix_pathfiles)
    pathfiles = check_files(prefix_pathfiles, result_files)
    print ('# of json files: ', len(pathfiles))
    #samples_pathfile = d['channel_samples_files'][0]
    samples_pathfile = d['test_channel_samples_files'][0]
    #num_of_levels_opts = d['num_of_levels_opts']

    #gridsize = (2, 3)
    #fig = plt.figure(figsize=(8, 12))
    n = int(sys.argv[1])

    rx_array_size = n
    tx_array_size = n

    initial_alphabet_opt = str(sys.argv[2])
    
    cb_est_dict = {}
    cb_est_num_of_levels_dict = {}
    count = 0
    for pathfile_id, pathfile in pathfiles.items():

        with open(pathfile) as result_pathfile:
            data = result_pathfile.read()
            d_result = json.loads(data)

        pass
        if d_result['rx_array_size'] == rx_array_size and d_result['tx_array_size'] == tx_array_size and d_result['initial_alphabet_opt'] == initial_alphabet_opt:
            pass
            count = count + 1
            print (f'pathfile: {pathfile}') 
            cb_est = get_codebook(pathfile)
            for k, v in cb_est.items():
                print (k)
    
