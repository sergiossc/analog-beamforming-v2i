import numpy as np
from numpy.linalg import svd, matrix_rank, norm
import matplotlib.pyplot as plt
from utils import * #richscatteringchnmtx, decode_codebook
import sys
import json
import pandas as pd
import os
from matplotlib.ticker import MaxNLocator
import uuid


if __name__ == "__main__":

    
    n_set = [4, 8, 16, 32, 64]
    #n_set = [64]

    #initial_alphabet_set = ['katsavounidis', 'xiaoxiao', 'random_from_samples', 'random']
    initial_alphabet_set = ['xiaoxiao', 'random_from_samples', 'random']

    num_levels_set = [4, 8, 16, 32, 64, 128, 256, 512]

    #dataset_name_set = ['s000', 's002', 's004', 's006', 's007', 's008', 's009']
    dataset_name_set = ['s002']


    num_of_trials =  10 #1000 # num of samples from test dataset
    
    #print (len(user_filter_set))
    # Read all json result files from training
    result_pathfiles_dict = get_all_result_json_pathfiles()
    print (len(result_pathfiles_dict))


    # Now I have to deal with each JSON result data file
    result_filter_set = {}
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

        result_filter = {'ds_name': ds_name, 'rx_array_size': rx_array_size, 'tx_array_size': tx_array_size, 'initial_alphabet_opt': initial_alphabet, 'num_of_levels': num_of_levels}
        result_filter_set[k] = result_filter

    # Now, hit it! 
    for dataset_name in dataset_name_set:
        #data_test_results = {}
        #data_test_results['dataset_name'] = dataset_name
        for n in n_set:
            print (f'ds_name: {dataset_name}\t-->\tn: {n}')
            #data_test_results['n'] = n
            test_channels_pathfile = f'/home/snow/analog-beamforming-v2i/{dataset_name}-test_set_{n}x{n}_a.npy'
            #data_test_results['test_channels_pathfile'] = test_channels_pathfile
            #print (f'loading data test channels in {test_channels_pathfile}... ')
            channels = np.load(f'{test_channels_pathfile}')
            #print (np.shape(channels))
            n_samples, n_rx, n_tx = np.shape(channels)
            #data_test_results['num_of_trials'] = num_of_trials
            my_seed = 5678
            np.random.seed(5678)
            #data_test_results['test_seed'] = my_seed
            if num_of_trials == -1:
                num_of_trials = n_samples
            ch_id_list = np.random.choice(len(channels), num_of_trials, replace=False)

            # Current dataset is 's000'
            # Current n is 'n'

            #dft_oversampling_factor = -1
            #bf_gain_dft_list_dict = {}

            # Create filter of current trial: user_filter_set = {}
            for num_levels in num_levels_set:
                print (f'num_levels: {num_levels}')
                #data_test_results[num_levels] = {} 
                #for num_levels in num_levels_set:
                current_user_filter_set = {}
                for initial_alphabet in initial_alphabet_set:
                    user_filter_id = uuid.uuid4()
                    user_filter = {'ds_name': dataset_name, 'rx_array_size': n, 'tx_array_size': n, 'initial_alphabet_opt': initial_alphabet, 'num_of_levels': num_levels}
                    current_user_filter_set[user_filter_id] = user_filter

                    #data_test_results[num_levels][initial_alphabet] = {}
        
                codebooks_from_results_json_dict = {}

                for filter_id, user_filter in current_user_filter_set.items():
                    for result_filter_id, result_filter in result_filter_set.items():
                        if compare_filter(user_filter, result_filter):
                            pass
                            codebooks_from_results_json_dict[filter_id] = result_pathfiles_dict[result_filter_id]
                pass
                cb_set = {} 
                cb_egt_set = {} 
                for k, v in codebooks_from_results_json_dict.items():
                    #print (f'k: {k}')
                    #print (f'v: {v}')
                    with open(v) as result:
                       result_data = result.read()
                       results_d = json.loads(result_data)
                    # Getting estimated codebook from VQ training
                    cb_dict = decode_codebook(results_d['codebook']) # no limited version of 'cb'
                    cb_set[k] = cb_dict
                    cb_egt_dict = {} # EGT version of the same 'cb'
                    for cw_id, cw in cb_dict.items():
                        cb_egt_dict[cw_id] = 1/(np.sqrt(n_tx)) * np.exp(1j * np.angle(cw))
                    cb_egt_set[k] = cb_egt_dict
 
                    #gain_est, cw_id_tx = beamsweeping2(ch, cb_dict)
                    #bf_gain_est[k].append(gain_est)
                    #bf_vec_est[k].append(cb_dict[cw_id_tx])



                # DFT
                cb_dft = None
                bf_gain_dft_list_dict = {}
                dft_oversampling_factor = int(num_levels/n)
                if (dft_oversampling_factor > 0): #
                    fftmat, ifftmat = fftmatrix(n_tx, dft_oversampling_factor)
                    cb_dft = matrix2dict(fftmat)
                    #if len(cb_dft) <= 512:
                    #print (f'dft_oversampling_factor: {dft_oversampling_factor}\nCB_DFT --> {len(cb_dft)}')
                    for k, v in cb_dft.items():
                        cb_dft[k] = v.T
                    bf_gain_dft_list_dict[len(cb_dft)] = []
                    #print (np.shape(v.T))
                    #print (f'---------------------------------> {len(cb_dft)}')


                # Current dataset is s000, current n is 'n' and current level is 'num_levels'

                bf_gain_max_list = []
                bf_gain_max_egt_list = []

                #bf_gain_dft_list_dict = {}  # This is a dict with a list of dft gain values where the key is the number of dft cw
                bf_gain_list_dict = {}
                bf_gain_egt_list_dict = {}
                for k in codebooks_from_results_json_dict.keys():
                    bf_gain_list_dict[k] = []
                    bf_gain_egt_list_dict[k] = []

                # Test really starts HERE!

                for ch_id in ch_id_list:
                    ch = channels[ch_id]
                    num_rx = np.shape(ch)[0]
                    num_tx = np.shape(ch)[1]
                    ch = ch/norm(ch)
                    ch = np.sqrt(num_rx * num_tx) * ch # meaning that squared norm is num_rx times num_tx

                    # BF gain max is given by first vector from SDV
                    u, s, vh = svd(ch)    # singular values 
                    f = np.matrix(vh[0,:]).T
                    w = np.matrix(u[:,0]).T
                    #print (f'f.shape: {np.shape(f)}')
                    #print (f'w.shape: {np.shape(w)}')
        
                    bf_gain_max = norm(w.conj().T * (ch * f.conj())) ** 2
                    bf_gain_max_info = {'bf_gain': bf_gain_max, 'f': f}
                    bf_gain_max_list.append(bf_gain_max_info)

                    # BF gain max using EGT
                    f_egt = 1/(np.sqrt(num_tx)) * np.exp(1j * np.angle(f)) # EGT version of 'f' from SVD
                    bf_gain_max_egt = norm(w.conj().T * (ch * f_egt.conj())) ** 2
                    bf_gain_max_egt_info = {'bf_gain': bf_gain_max_egt, 'f': f_egt}
                    bf_gain_max_egt_list.append(bf_gain_max_egt_info)
                    
                    if (dft_oversampling_factor > 0): #512:
                        bf_gain_dft, cw_id_tx_dft = beamsweeping2(ch, cb_dft)
                        bf_gain_dft_info = {'bf_gain': bf_gain_dft, 'f': cb_dft[cw_id_tx_dft]}
                        bf_gain_dft_list_dict[len(cb_dft)].append(bf_gain_dft_info)

                    for k, cb_dict in cb_set.items():
                        bf_gain, cw_id_tx = beamsweeping2(ch, cb_dict)
                        bf_gain_info = {'bf_gain': bf_gain, 'f': cb_dict[cw_id_tx]}
                        bf_gain_list_dict[k].append(bf_gain_info)
                    for k, cb_egt_dict in cb_egt_set.items():
                        bf_gain_egt, cw_id_tx = beamsweeping2(ch, cb_egt_dict)
                        bf_gain_egt_info = {'bf_gain': bf_gain_egt, 'f': cb_egt_dict[cw_id_tx]}
                        bf_gain_egt_list_dict[k].append(bf_gain_egt_info)
                
                # Ended this round with this num_levels, take notos from data
                data_to_save = {}
                data_to_save['dataset_name'] = dataset_name
                data_to_save['n'] = n
                data_to_save['test_channels_pathfile'] = test_channels_pathfile
                data_to_save['num_of_test_samples'] = num_of_trials
                data_to_save['my_seed'] = my_seed
                data_to_save['num_levels'] = num_levels
                data_to_save['bf_gain_max_list'] = [v['bf_gain'] for v in bf_gain_max_list]
                data_to_save['cw_gain_max_list'] = encode_codebook(matrix2dict([v['f'] for v in bf_gain_max_list]))
                data_to_save['bf_gain_max_egt_list'] = [v['bf_gain'] for v in bf_gain_max_egt_list]
                data_to_save['cw_gain_max_egt_list'] = encode_codebook(matrix2dict([v['f'] for v in bf_gain_max_egt_list]))
                ##data_to_save['vec_gain_max_list'] = [v['f'] for v in bf_gain_max_list]


                for k in codebooks_from_results_json_dict.keys():
                    initial_alphabet = current_user_filter_set[k]['initial_alphabet_opt']
                    data_to_save[initial_alphabet] = {}
                for k in codebooks_from_results_json_dict.keys():
                    initial_alphabet = current_user_filter_set[k]['initial_alphabet_opt']
                    data_to_save[initial_alphabet]['bf_gain_list'] = [v['bf_gain'] for v in bf_gain_list_dict[k]]
                    data_to_save[initial_alphabet]['f'] = encode_codebook(matrix2dict([v['f'] for v in bf_gain_list_dict[k]]))
                    data_to_save[initial_alphabet]['bf_gain_egt_list'] = [v['bf_gain'] for v in bf_gain_egt_list_dict[k]]
                    data_to_save[initial_alphabet]['f_egt'] = encode_codebook(matrix2dict([v['f'] for v in bf_gain_egt_list_dict[k]]))



                if (dft_oversampling_factor > 0): #512:
                    for k, v in bf_gain_dft_list_dict.items():
                        data_to_save['dft'] = {}
                    for k, v in bf_gain_dft_list_dict.items():
                        data_to_save['dft']['num_of_levels'] = k
                        data_to_save['dft']['bf_gain_list'] = [i['bf_gain'] for i in v]
                        data_to_save['dft']['f'] = encode_codebook(matrix2dict([i['f'] for i in v]))
                #pass
                #json_filename = f'test_results/test-result-{dataset_name}-n{n}-l{num_levels}.json'
                json_filename = f'test_results_all_test_samples/test-result-{dataset_name}-n{n}-l{num_levels}.json'
                with open(json_filename, "w") as write_file:
                    json.dump(data_to_save, write_file, indent=4)
                pass

            pass
 
###
