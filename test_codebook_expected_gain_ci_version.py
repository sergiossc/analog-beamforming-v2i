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

    initial_alphabet_set = ['katsavounidis', 'xiaoxiao', 'random_from_samples', 'random']
    #initial_alphabet_set_label_pt_dict = {'katsavounidis': 'katsavounidis', 'xiaoxiao':'xiaoxiao', 'random_from_samples':'aleatório(amostras)', 'random':'aleatório'}
    #num_levels_set = [4, 8, 16, 32, 64, 128, 256, 512]
    num_levels_set = [4, 8]
    dataset_name = 's000'

    #for n_level in num_levels_set:

    for n in n_set:
        #fig, ax = plt.subplots(figsize=(20, 10))
        #major_user_filter_set = {}
        bf_gain_mean = {}

        for num_levels in num_levels_set:
            bf_gain_mean[num_levels] = {}
            for initial_alphabet in initial_alphabet_set:
                bf_gain_mean[num_levels][initial_alphabet] = -1


        for num_levels in num_levels_set:
            #for num_levels in num_levels_set:
            user_filter_set = {}
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
                        #print (f'user filter is {user_filter}')
                        #print (f'result filter is {result_filter}\n')
                        #print (f'{pathfile}') 
                        codebooks_from_results_json_dict[filter_id] = pathfile
     
    
            pass
            test_channels_pathfile = f'/home/snow/analog-beamforming-v2i/{dataset_name}-test_set_{n}x{n}_a.npy'
            print (test_channels_pathfile)
            channels = np.load(f'{test_channels_pathfile}')
            print (np.shape(channels))
            n_samples, n_rx, n_tx = np.shape(channels)
            num_of_trials = 5
            np.random.seed(5678)
            ch_id_list = np.random.choice(len(channels), num_of_trials, replace=False)
    
            #bf_gain_opt = {}
            bf_gain_opt = []
            #bf_gain_opt['egt'] = []

            # DFT
            fftmat, ifftmat = fftmatrix(n_tx, None)
            cb_dft = matrix2dict(fftmat)
            for k, v in cb_dft.items():
                cb_dft[k] = v.T
            bf_gain_dft = []
    
            #bf_vec_opt = {}
            bf_vec_opt = []
            #bf_vec_opt['egt'] = []
    
            bf_gain_est = {}
            bf_vec_est = {}
            # k is the filter_id
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
                f = 1/(np.sqrt(num_tx)) * np.exp(1j * np.angle(f))
                gain_opt = norm(w.conj().T * (ch * f.conj())) ** 2
                bf_gain_opt.append(gain_opt)
                bf_vec_opt.append(f)

                gain_dft, cw_id_tx_dft = beamsweeping2(ch, cb_dft)
                bf_gain_dft.append(gain_dft)
 
                for k, v in codebooks_from_results_json_dict.items():
                    #print (f'k: {k}')
                    #print (f'v: {v}')
                    with open(v) as result:
                       result_data = result.read()
                       results_d = json.loads(result_data)
                    # Getting estimated codebook from VQ training
                    cb_dict = decode_codebook(results_d['codebook'])
                    for cw_id, cw in cb_dict.items():
                        cb_dict[cw_id] = 1/(np.sqrt(num_tx)) * np.exp(1j * np.angle(cw))

                    gain_est, cw_id_tx = beamsweeping2(ch, cb_dict)
                    bf_gain_est[k].append(gain_est)
                    bf_vec_est[k].append(cb_dict[cw_id_tx])
    


            #bf_gain_mean[num_levels]['opt'] = np.mean(bf_gain_opt)
            bf_gain_mean[num_levels]['opt'] = bf_gain_opt
            #bf_gain_mean[num_levels]['dft'] = np.mean(bf_gain_dft)
            bf_gain_mean[num_levels]['dft'] = bf_gain_dft
            #bf_gain_mean[num_levels]['opt'] = bf_gain_opt
            #print (len(bf_gain_opt))

            for k, v in codebooks_from_results_json_dict.items():
                #bf_gain_mean[num_levels][user_filter_set[k]['initial_alphabet_opt']] = np.mean(bf_gain_est[k])
                bf_gain_mean[num_levels][user_filter_set[k]['initial_alphabet_opt']] = bf_gain_est[k]

        #plt.show()
        fig, ax = plt.subplots(figsize=(10, 10))

        marker_dict = {'katsavounidis': 8, 'random': 9, 'random_from_samples': 10, 'xiaoxiao': 11, 'opt': 5, 'dft':'+'}
        color_dict = {'dft': 'darkviolet', 'opt': 'black', 'random': 'green', 'random_from_samples': 'red', 'xiaoxiao': 'orange', 'katsavounidis': 'blue' }
        initial_alphabet_set_label_pt_dict = {'dft': 'DFT', 'opt': 'BF ideal(EGT)', 'katsavounidis': 'katsavounidis(EGT)', 'xiaoxiao':'xiaoxiao(EGT)', 'random_from_samples':'aleatório(amostras)(EGT)', 'random':'aleatório(EGT)'}
        sorted_cols = ['katsavounidis', 'random', 'random_from_samples', 'xiaoxiao']
        sorted_cols_pt = [initial_alphabet_set_label_pt_dict[v] for v in sorted_cols]

        color_list = [v for v in color_dict.values()]

        #print (bf_gain_mean)
        #df = pd.DataFrame(bf_gain_mean).T
        #print (df)

        #ax.plot(range(len(num_levels_set)), np.array(df['opt']), color=color_dict['opt'], label=initial_alphabet_set_label_pt_dict['opt'], marker=marker_dict['opt'], linestyle='dotted')

        #for df_col in df.columns:
        #    pass
        #    print (df_col)
        #    print (np.array(df[df_col]))
        #    if df_col == 'dft' or df_col == 'opt':
        #        pass
        #    else:
        #        ax.plot(range(len(num_levels_set)), np.array(df[df_col]), color=color_dict[df_col], label=initial_alphabet_set_label_pt_dict[df_col], marker=marker_dict[df_col], linestyle='dotted')

        #ax.plot(range(len(num_levels_set)), np.array(df['dft']), color=color_dict['dft'], label=initial_alphabet_set_label_pt_dict['dft'], marker=marker_dict['dft'])

        conf_level = 0.90
        data = np.zeros(num_of_trials)
        conf_intervals = []
        for l in num_levels_set:
            for initial_alpha in initial_alphabet_set:
                sequence_values = np.array(bf_gain_mean[l][initial_alpha])
                data = np.column_stack((data, sequence_values))
                sample_mean, ci_values = conf_interval(sequence_values, conf_level)
                conf_intervals.append([ci_values])
        ax.boxplot(data.T, conf_intervals=conf_intervals, notch=True)

        ##ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
        #ax.set_ylim(0,16)
        #plt.xticks(range(len(num_levels_set)), np.array(num_levels_set))
        ##plt.xlabel('Tamanho do codebook (L)', fontsize=11)
        ##plt.ylabel(r'Ganho médio de BF ($\mathbb{E}(G)$)', fontsize=11)
        ##plt.grid(True)
        ##image_filename = f'test-bf-expected-gain-{dataset_name}-n{n}.png'
        #print (image_filename)
        #plt.savefig(image_filename, bbox_inches='tight')
        plt.show()
