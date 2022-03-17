import numpy as np
from numpy.linalg import svd, matrix_rank, norm
import matplotlib.pyplot as plt
from utils import * #richscatteringchnmtx, decode_codebook
import sys
import json
import pandas as pd
import os
from matplotlib.ticker import MaxNLocator



if __name__ == "__main__":

    
    #n_set = [4, 8, 16, 32, 64]
    n_set = [4]
    #setup_dict = {4:'I', 8:'II', 16:'III', 32:'IV', 64:'V'}
    #small_axis_y_lim_dict = {4:-45, 8:-40, 16:-35, 32:-35, 64:-33}

    initial_alphabet_set = ['random_from_samples', 'random']
    #initial_alphabet_set = ['katsavounidis', 'xiaoxiao', 'random_from_samples', 'random']
    initial_alphabet_set_label_pt_dict = {'random_from_samples':'aleatório(amostras)', 'random':'aleatório'}
    #num_levels_set = [4, 8, 16, 32, 64, 128, 256, 512]
    num_levels_set = [4, 8, 16]
    dataset_name = 's008'

    #for n_level in num_levels_set:

    for n in n_set:
        
        for num_levels in num_levels_set:
            pass


        for num_levels in num_levels_set:
            user_filter_set = {}
            #for num_levels in num_levels_set:
            for initial_alphabet in initial_alphabet_set:
                user_filter_id = uuid.uuid4()
                user_filter = {'ds_name': dataset_name, 'rx_array_size': n, 'tx_array_size': n, 'initial_alphabet_opt': initial_alphabet, 'num_of_levels': num_levels}
                user_filter_set[user_filter_id] = user_filter
    
            codebooks_from_results_json_dict = {}
            for user_filter_id in user_filter_set.keys():
                codebooks_from_results_json_dict[user_filter_id] = ''
            #    min_mean_distortion_dict[filter_id] = None
        
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
                        codebooks_from_results_json_dict[filter_id] = pathfile
    
            pass
            test_channels_pathfile = f'/home/snow/analog-beamforming-v2i/{dataset_name}-test_set_{n}x{n}_a.npy'
            print (test_channels_pathfile)
            channels = np.load(f'{test_channels_pathfile}')
            print (np.shape(channels))
            n_samples, n_rx, n_tx = np.shape(channels)
            num_of_trials = 10
            np.random.seed(5678)
            ch_id_list = np.random.choice(len(channels), num_of_trials, replace=False)
    
            bf_gain_opt = {}
            bf_gain_opt['opt'] = []
            #bf_gain_opt['egt'] = []
    
            bf_vec_opt = {}
            bf_vec_opt['opt'] = []
            #bf_vec_opt['egt'] = []
    
            bf_gain_est = {}
            bf_vec_est = {}
            for k in codebooks_from_results_json_dict.keys():
                bf_gain_est[k] = []
                bf_vec_est[k] = []
    
    
            for ch_id in ch_id_list:
                ch = channels[ch_id]
                num_rx = np.shape(ch)[0]
                num_tx = np.shape(ch)[1]
                ch = ch/norm(ch)
                ch = np.sqrt(num_rx * num_tx) * ch # meaning that squared norm is num_rx times num_tx
                
                u, s, vh = svd(ch)    # singular values 
                f = np.matrix(vh[0,:]).T
                w = np.matrix(u[:,0]).T
                #print (f'f.shape: {np.shape(f)}')
                #print (f'w.shape: {np.shape(w)}')
    
                #gain_opt = norm(w.conj().T * (ch * f.conj().T)) ** 2
                gain_opt = norm(w.conj().T * (ch * f.conj())) ** 2
                bf_gain_opt['opt'].append(gain_opt)
                bf_vec_opt['opt'].append(f)
    
                for k, v in codebooks_from_results_json_dict.items():
                    print (f'k: {k}')
                    print (f'v: {v}')
                    with open(v) as result:
                       result_data = result.read()
                       results_d = json.loads(result_data)
                    # Getting estimated codebook from VQ training
                    cb_dict = decode_codebook(results_d['codebook'])
                    gain_est, cw_id_tx = beamsweeping2(ch, cb_dict)
                    bf_gain_est[k].append(gain_est)
                    bf_vec_est[k].append(cb_dict[cw_id_tx])
    
            #count_marker = 0
            #markers_list = ["*", "h", "p", 4, 5, 6, 7, 8, 9, 10, 11, "+", "."]
            color_dict = {'random': 'green', 'random_from_samples': 'red'}
            marker_dict = {4: "*", 8: "h", 16: "p", 32: 4, 64:5, 128:6, 256:7, 512:10}
            initial_alphabet_set_label_pt_dict = {'random_from_samples':'aleatório(amostras)', 'random':'aleatório'}
    
            fig, ax = plt.subplots(figsize=(20, 10))
    
            ax.plot(bf_gain_opt['opt'], label='Ganho de BF')
            #ax.plot(bf_gain_opt['opt'], marker=markers_list[count_marker], label='BF gain')
            #ax.plot(bf_gain_opt['egt'], linestyle='dotted', color=plt.gca().lines[-1].get_color(), marker=markers_list[count_marker], label='BF gain (EGT)')
            #count_marker += 1
    
    
            ordered_keys_by_initial_alphabet = {'random_from_samples': '', 'random': ''}
            for k, v in bf_gain_est.items():
                print (k)
                init_alph = user_filter_set[k]['initial_alphabet_opt']
                print (init_alph)
                ordered_keys_by_initial_alphabet[init_alph] =  k
             
               
            print (ordered_keys_by_initial_alphabet)
            for k in ordered_keys_by_initial_alphabet.values():
                v = bf_gain_est[k]
                init_alph = user_filter_set[k]['initial_alphabet_opt']
                init_alph_pt = initial_alphabet_set_label_pt_dict[init_alph]
                l = user_filter_set[k]['num_of_levels']
                ax.plot(np.arange(num_of_trials), v, marker=marker_dict[l], linestyle='dashed', color=color_dict[init_alph], label=f'Ganho de BF ({init_alph_pt}), L={l}')
                #ax.plot(np.arange(num_of_trials), v, marker=markers_list[count_marker], color=color_dict[init_alph], label=f'Ganho de BF ({init_alph_pt}), L={l}')
                #ax.plot(np.arange(num_of_trials), bf_gain_est_egt[k], linestyle='dotted', color=plt.gca().lines[-1].get_color(), marker=markers_list[count_marker], label=f'BF gain est (EGT), L={k}')
                #count_marker += 1
            #fig, ax = plt.subplots(figsize=(20,10)) # (x_size, y_size)
            #ax.plot(np.arange(num_of_trials), bf_gain_dft, marker=markers_list[count_marker], label=f'DFT gain, L={n_tx}')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend(loc='best', fontsize=11)
            plt.xlabel('Realizações de canal', fontsize=11)
            plt.ylabel(r'Ganho de beamforming ($|w^{H}_{opt} \times (H \times f^{*}|^{2})$', fontsize=11)
            #plt.title(f'{test_channels_pathfile}')
            plt.grid(True)
            image_filename = f'test_bf_gain_{dataset_name}-l{num_levels_set}-n{n}.png'
            print (image_filename)
            #plt.savefig(image_filename, bbox_inches='tight')
            plt.show()
    
    #-----------------------------------------------------x
