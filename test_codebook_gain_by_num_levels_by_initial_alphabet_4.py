import numpy as np
from numpy.linalg import svd, matrix_rank, norm
import matplotlib.pyplot as plt
from utils import * #richscatteringchnmtx, decode_codebook
import sys
import json
import pandas as pd
import os
from matplotlib.ticker import MaxNLocator


plt.rcParams.update({'font.size':9})
plt.rcParams['text.usetex'] = True


if __name__ == "__main__":


    #dataset_name_set = ['s000', 's002', 's004', 's006', 's007', 's008', 's009']
    #dataset_name_set = ['s002', 's004', 's006', 's007', 's008', 's009']
    dataset_name_set = ['s002']


    for dataset_name in dataset_name_set:
        
        #n_set = [4, 8, 16, 32, 64]
        n_set = [4]
        #setup_dict = {4:'I', 8:'II', 16:'III', 32:'IV', 64:'V'}
    
        initial_alphabet_set = ['xiaoxiao', 'random_from_samples', 'random']
        #initial_alphabet_set_label_pt_dict = {'katsavounidis': 'katsavounidis', 'xiaoxiao':'xiaoxiao', 'random_from_samples':'aleatório(amostras)', 'random':'aleatório'}
        num_levels_set = [4, 8, 16, 32, 64, 128, 256, 512]
        #num_levels_set = [4, 8]
        #dataset_name = 's000'
    
        #for n_level in num_levels_set:

    
        for n in n_set:
            #fig, ax = plt.subplots(figsize=(20, 10))
            #major_user_filter_set = {}
            bf_gain = {}
            bf_gain_egt = {}
            bf_gain_dft_dict = {}
            #fig, ax = plt.subplots(1,2, figsize=(10, 10))
            fig, ax = plt.subplots(figsize=(20, 10))
            #fig, ax = plt.subplots(1,2, figsize=(20,10), gridspec_kw = {'wspace':0.025, 'hspace':0.15}, sharey=True)
            num_of_trials = 50
    
            for num_levels in num_levels_set:
                bf_gain[num_levels] = {}
                bf_gain_egt[num_levels] = {}
                bf_gain_dft_dict[num_levels] = {}
                for initial_alphabet in initial_alphabet_set:
                    bf_gain[num_levels][initial_alphabet] = []
                    bf_gain_egt[num_levels][initial_alphabet] = []
                    bf_gain_dft_dict[num_levels][initial_alphabet] = []
            #bf_gain[]['opt'] = []
    
    
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
            
                #result_pathfiles_dict = get_all_result_json_pathfiles()
                result_pathfiles_dict = get_all_result_json_pathfiles(rootdir="results/s002")
            
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
                test_channels_pathfile = f'/home/snow/analog-beamforming-v2i/{dataset_name}-test_set_{n}x{n}_normalized.npy'
                print (test_channels_pathfile)
                channels = np.load(f'{test_channels_pathfile}')
                print (np.shape(channels))
                n_samples, n_rx, n_tx = np.shape(channels)
                #num_of_trials = 50
                np.random.seed(5678)
                #np.random.seed(910)
                ch_id_list = np.random.choice(len(channels), num_of_trials, replace=False)
        
                #bf_gain_opt = {}
                bf_gain_opt = []
                bf_gain_opt_egt = []
        
                #bf_vec_opt = {}
                bf_vec_opt = []
                #bf_vec_opt['egt'] = []
        
                bf_gain_est = {}
                bf_gain_est_egt = {}
    
                bf_vec_est = {}
                # k is the filter_id
                for k in codebooks_from_results_json_dict.keys():
                    bf_gain_est[k] = []
                    bf_gain_est_egt[k] = []
                    bf_vec_est[k] = []
    
    
                # DFT
                fftmat, ifftmat = fftmatrix(n_tx, None)
                cb_dft = matrix2dict(fftmat)
                for k, v in cb_dft.items():
                    cb_dft[k] = v.T
                bf_gain_dft = []
                
        
                for ch_id in ch_id_list:
                    ch = channels[ch_id]
                    num_rx = np.shape(ch)[0]
                    num_tx = np.shape(ch)[1]
                    ##ch = ch/norm(ch)
                    ##ch = np.sqrt(num_rx * num_tx) * ch # meaning that squared norm is num_rx times num_tx
                    
                    u, s, vh = svd(ch)    # singular values 
                    f = np.matrix(vh[0,:]).T
                    w = np.matrix(u[:,0]).T
                    #print (f'------> f.norm: {norm(f)}')
                    #print (f'------> w.norm: {norm(w)}')
        
                    #gain_opt = norm(w.conj().T * (ch * f.conj().T)) ** 2
                    gain_opt = norm(w.conj().T * (ch * f.conj())) ** 2
                    bf_gain_opt.append(gain_opt)
    
                    f_egt = 1/(np.sqrt(num_tx)) * np.exp(1j * np.angle(f))
                    gain_opt_egt = norm(w.conj().T * (ch * f_egt.conj())) ** 2
                    bf_gain_opt_egt.append(gain_opt_egt)
                    #print (f'------> f_egt.norm : {norm(f_egt)}')
    
                    gain_dft, cw_id_tx_dft = beamsweeping2(ch, cb_dft)
                    bf_gain_dft.append(gain_dft)
    
                    #bf_vec_opt.append(f)
        
                    for k, v in codebooks_from_results_json_dict.items():
                        #print (f'k: {k}')
                        #print (f'v: {v}')
                        with open(v) as result:
                           result_data = result.read()
                           results_d = json.loads(result_data)
                        # Getting estimated codebook from VQ training
                        cb_dict = decode_codebook(results_d['codebook'])
    
                        gain_est, cw_id_tx = beamsweeping2(ch, cb_dict)
                        bf_gain_est[k].append(gain_est)
                        #bf_vec_est[k].append(cb_dict[cw_id_tx])
        
                        cb_dict_egt = {}
                        for cw_id, cw in cb_dict.items():
                            cb_dict_egt[cw_id] = 1/(np.sqrt(num_tx)) * np.exp(1j * np.angle(cw))
                            print (f'------> cw.norm: {norm(cw)}')
                            print (f'------> cw_egt.norm: {norm(cb_dict_egt[cw_id])}')
                        gain_est_egt, cw_id_tx_egt = beamsweeping2(ch, cb_dict_egt)
                        bf_gain_est_egt[k].append(gain_est_egt)
                        #bf_vec_est_egt[k].append(cb_dict_egt[cw_id_tx_egt])
    
    
                #bf_gain['opt']['opt'] = bf_gain_opt
                bf_gain[num_levels]['opt'] = bf_gain_opt
                bf_gain_egt[num_levels]['egt'] = bf_gain_opt_egt
                bf_gain_dft_dict[num_levels]['dft'] = bf_gain_dft
                #print (len(bf_gain_opt))
    
                for k, v in codebooks_from_results_json_dict.items():
                    bf_gain[num_levels][user_filter_set[k]['initial_alphabet_opt']] = bf_gain_est[k]
                    bf_gain_egt[num_levels][user_filter_set[k]['initial_alphabet_opt']] = bf_gain_est_egt[k]
                    #bf_gain_mean[num_levels][user_filter_set[k]['initial_alphabet_opt']] = bf_gain_est[k]
    
            #plt.show()
            # PLOTTING STUFF
    
            #marker_dict = {'katsavounidis': "*", 'random': "h", 'random_from_samples': "p", 'xiaoxiao': 4, 'opt': 5}
            marker_dict = {4: "*", 8: "h", 16: "p", 32: 4, 64:5, 128:6, 256:7, 512:10}
            color_dict = {'dft': 'darkviolet', 'opt': 'black', 'random': 'green', 'random_from_samples': 'red', 'xiaoxiao': 'orange', 'katsavounidis': 'blue' }
            initial_alphabet_set_label_pt_dict = {'opt': 'Beamforming ideal', 'katsavounidis': 'katsavounidis', 'xiaoxiao':'xiaoxiao', 'random_from_samples':'aleatório(amostras)', 'random':'aleatório'}
            sorted_cols = ['xiaoxiao', 'random_from_samples', 'random']
            sorted_cols_pt = [initial_alphabet_set_label_pt_dict[v] for v in sorted_cols]
        
            color_list = [v for v in color_dict.values()]
        
            #df = pd.DataFrame(bf_gain).T
            #print (df)
        
        
            ##ax.plot(np.arange(num_of_trials), np.array(bf_gain[4]['opt']), color=color_dict['opt'], label=f"{initial_alphabet_set_label_pt_dict['opt']}", marker=11)
            ax.plot(np.arange(num_of_trials), np.array(bf_gain_egt[4]['egt']), color=color_dict['opt'], linestyle='dotted', label=f"{initial_alphabet_set_label_pt_dict['opt']} (EGT)", marker=11)

            data_stack = None
            data_stack_egt = None


            for l, v1 in bf_gain.items():
                for initial_alphabet, bf_gain_values in v1.items():
                    pass
                    if initial_alphabet == 'opt':
                        pass
                    else:
                        pass
                        a = np.matrix(bf_gain_values)
                        #print (np.shape(a))
                        #print (np.shape(a_egt))

                        if data_stack is None:
                            data_stack = a
                        else:
                            data_stack = np.vstack((data_stack, a))

                        print ('*******************')



            for l, v1 in bf_gain_egt.items():
                for initial_alphabet, bf_gain_values in v1.items():
                    pass
                    #print (l)
                    #print (initial_alphabet)
                    #print (bf_gain_values)
                    #print (np.array(df[df_col]))
                    if initial_alphabet == 'opt':
                        pass
                    else:
                        pass
                        a_egt = np.matrix(bf_gain_egt[l][initial_alphabet])
                        #print (np.shape(a))
                        #print (np.shape(a_egt))

                        if data_stack_egt is None:
                            data_stack_egt = a_egt
                        else:
                            data_stack_egt = np.vstack((data_stack_egt, a_egt))
       
                        print ('*******************')
            #            ax.plot(np.arange(num_of_trials), np.array(bf_gain_values), color=color_dict[initial_alphabet], label=f'{initial_alphabet_set_label_pt_dict[initial_alphabet]}, L={l}', marker=marker_dict[l])
            #            ax.plot(np.arange(num_of_trials), np.array(bf_gain_egt[l][initial_alphabet]), linestyle='dotted', color=color_dict[initial_alphabet], label=f'{initial_alphabet_set_label_pt_dict[initial_alphabet]}(EGT), L={l}', marker=marker_dict[l])
            #print ('BEGIN --- xxxxxxxxxxxxxxxxxxxxxxx')
            #print (np.shape(data_stack))
            data_stack_mean = np.mean(data_stack, axis=0)
            data_stack_mean_n = np.array(data_stack_mean).reshape(num_of_trials)
            #print (data_stack_mean_n)
            #print (np.shape(data_stack_mean_n))
            #ax.plot(np.arange(num_of_trials), data_stack_mean.T, label='mean est')
            data_stack_std = np.std(data_stack, axis=0)
            data_stack_std_n = np.array(data_stack_std).reshape(num_of_trials)
            #print (data_stack_std_n)
            #print (np.shape(data_stack_std_n))
            x = np.arange(num_of_trials)
            #print (x)
            ##ax.errorbar(x, data_stack_mean_n, yerr=data_stack_std_n, capsize=5, label=r'$\mu_{G}$ e $\sigma_{G}$ dos codebooks estimados para cada realização \textbf{H}', marker='o', color='green')
            ##ax.plot(x, np.ones(num_of_trials)*np.mean(data_stack_mean_n), label=r'$\mu_{G}$ dos codebooks estimados para o conjunto de 50 realizações de \textbf{H}', color='green')

            ##ax.plot(np.arange(num_of_trials), np.ones(num_of_trials)*np.mean(np.array(bf_gain[4]['opt'])), color=color_dict['opt'], label=r'$\mu_{G}$'+ f" do {initial_alphabet_set_label_pt_dict['opt']}" + r' para 50 realizações de \textbf{H}')
            #print ('END --- xxxxxxxxxxxxxxxxxxxxxxx')
        
            #print ('BEGIN --- xxxxxxxxxxxxxxxxxxxxxxx')
            #print (np.shape(data_stack))
            data_stack_mean_egt = np.mean(data_stack_egt, axis=0)
            data_stack_mean_egt_n = np.array(data_stack_mean_egt).reshape(num_of_trials)
            #print (data_stack_mean_egt_n)
            #print (np.shape(data_stack_mean_egt_n))
            #ax.plot(np.arange(num_of_trials), data_stack_mean.T, label='mean est')
            data_stack_std_egt = np.std(data_stack_egt, axis=0)
            data_stack_std_egt_n = np.array(data_stack_std_egt).reshape(num_of_trials)
            #print (data_stack_std_egt_n)
            #print (np.shape(data_stack_std_egt_n))
            x = np.arange(num_of_trials)
            #print (x)
            ax.errorbar(x, data_stack_mean_egt_n, yerr=data_stack_std_egt_n, capsize=5, label=r'$\mu_{G}$ e $\sigma_{G}$ dos codebooks estimados~(EGT) para cada realização \textbf{H}', marker='o', linestyle='dotted', color='red')
            ##ax.plot(x, np.ones(num_of_trials)*np.mean(data_stack_mean_egt_n), label=r'$\mu_{G}$ com EGT dos codebooks estimados para 50 realizações de \textbf{H}', linestyle='dotted', color='red')

            ##ax.plot(np.arange(num_of_trials), np.ones(num_of_trials)*np.mean(np.array(bf_gain_egt[4]['egt'])), color=color_dict['opt'], linestyle='dotted', label=r'$\mu_{G}$'+f" do {initial_alphabet_set_label_pt_dict['opt']}"+r'~(EGT) para 50 realizações de \textbf{H}')
            #print ('END --- xxxxxxxxxxxxxxxxxxxxxxx')

            ##ax[0].plot(np.arange(num_of_trials), np.array(bf_gain_dft_dict[4]['dft']), color=color_dict['dft'], label='DFT', marker='+')
            ##ax[1].plot(np.arange(num_of_trials), np.array(bf_gain_dft_dict[4]['dft']), color=color_dict['dft'], label='DFT', marker='+')
        
            ax.legend(loc='best', fontsize=14)
            ax.legend(loc='best', fontsize=14)
            ax.grid(True)
            ax.grid(True)
            #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11, ncol=2)
            #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=11)
            #ax.set_ylim(0,16)
            #plt.xticks(np.arange(num_of_trials), np.array(num_of_trials))
            ##plt.xlabel(r'Realizações de canal (\textbf{H})', fontsize=9)
            ##plt.ylabel(r'Ganho de BF (G)', fontsize=9)
            ##plt.grid(True)
  
            #plt.xticks(np.arange(num_of_trials), np.arange(num_of_trials))
            plt.setp(ax, xticks=np.arange(num_of_trials) , xticklabels=np.arange(num_of_trials))

            fig.text(0.5, 0.05, r'Realizações de canal (\textbf{H})', ha='center', fontsize=14)
            fig.text(0.09, 0.5, r'Ganho de Beamforming (G)', va='center', rotation='vertical', fontsize=14)

            image_filename = f'test_bf_gain_{dataset_name}-vs-channel-realizations-n{n}_new.png'
            print (image_filename)
            plt.savefig(image_filename, bbox_inches='tight')
            #plt.show()
