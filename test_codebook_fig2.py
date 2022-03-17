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

    
    profile_pathfile_json = sys.argv[1]
    #result_pathfile_json = sys.argv[2]
    best_codebooks_from_results_json = sys.argv[2]

    #pathfile = result_pathfile_json

    with open(profile_pathfile_json) as profile:
        profile_data = profile.read()
        profile_d = json.loads(profile_data)

    test_channels_pathfile = profile_d['test_channel_samples_files'][0]
    print (test_channels_pathfile)

    channels = np.load(f'{test_channels_pathfile}')
    n_samples, n_rx, n_tx = np.shape(channels)
    print (np.shape(channels))

    # Getting DFT codebook
    fftmat, ifftmat = fftmatrix(n_tx, None)
    cb_dft = matrix2dict(fftmat)
    for k, v in cb_dft.items():
        cb_dft[k] = v.T
 


    num_of_trials = 50
    np.random.seed(5678)
    ch_id_list = np.random.choice(len(channels), num_of_trials, replace=False)

    with open(best_codebooks_from_results_json) as best_codebooks:
        best_codebooks_data = best_codebooks.read()
        best_results_d = json.loads(best_codebooks_data)



    bf_gain_opt = {}
    bf_gain_opt['opt'] = []
    bf_gain_opt['egt'] = []

    # DFT 
    bf_gain_dft = []


    bf_gain_est = {}
    for k in best_results_d.keys():
        bf_gain_est[k] = []

    bf_gain_est_egt = {}
    for k in best_results_d.keys():
        bf_gain_est_egt[k] = []

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
        f_egt = 1/(np.sqrt(num_tx)) * np.exp(1j * np.angle(f))
        gain_opt_egt = norm(w.conj().T * (ch * f_egt.conj())) ** 2
        bf_gain_opt['egt'].append(gain_opt_egt)

        gain_dft, cw_id_tx_dft = beamsweeping2(ch, cb_dft)
        #gain_dft = norm(w.conj().T * (ch * f_egt.conj())) ** 2
        bf_gain_dft.append(gain_dft)


        for k, v in best_results_d.items():
            #print (f'k: {k}, v:{v}')

            with open(v) as result:
               result_data = result.read()
               results_d = json.loads(result_data)
        

            # Getting estimated codebook from VQ training
            cb_dict = decode_codebook(results_d['codebook'])

            gain_est, cw_id_tx = beamsweeping2(ch, cb_dict)
            bf_gain_est[k].append(gain_est)

            cb_dict_egt = {}
            for cw_id, cw in cb_dict.items():
                cb_dict_egt[cw_id] = 1/(np.sqrt(num_tx)) * np.exp(1j * np.angle(cw))
            gain_est_egt, cw_id_tx_egt = beamsweeping2(ch, cb_dict_egt)
            bf_gain_est_egt[k].append(gain_est_egt)

    count_marker = 0
    markers_list = ["*", "h", "p", 4, 5, 6, 7, 8, 9, 10, 11, "+", "."]

    fig, ax = plt.subplots(figsize=(45, 5))

    ax.plot(bf_gain_opt['opt'], marker=markers_list[count_marker], label='BF gain')
    ax.plot(bf_gain_opt['egt'], linestyle='dotted', color=plt.gca().lines[-1].get_color(), marker=markers_list[count_marker], label='BF gain (EGT)')
    count_marker += 1


    for k, v in bf_gain_est.items():
        ax.plot(np.arange(num_of_trials), v, marker=markers_list[count_marker], label=f'BF gain est, L={k}')
        ax.plot(np.arange(num_of_trials), bf_gain_est_egt[k], linestyle='dotted', color=plt.gca().lines[-1].get_color(), marker=markers_list[count_marker], label=f'BF gain est (EGT), L={k}')
        count_marker += 1

    ax.plot(np.arange(num_of_trials), bf_gain_dft, marker=markers_list[count_marker], label=f'DFT gain, L={n_tx}')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='best')
    plt.xlabel('Realizações de canal', fontsize=20)
    plt.ylabel(r'Ganho de beamforming ($|w^{H}_{opt} \times (H \times f^{*}|^{2})$', fontsize=20)
    plt.title(f'{test_channels_pathfile}')
    plt.grid(True)
    plt.show()
##
##    for i in range(num_of_trials):
##        print (f'----------------------------------------------------------------------------x')
##        label_text = []
##
##        v_opt = bf_vector_opt[i]
##        label_text.append(f'opt')
##        print (f'norm(v_opt): {norm(v_opt)}')
##        print (f'abs(v_opt): \n{np.abs(v_opt)}')
##        print (f'angle(v_opt): \n{np.rad2deg(np.angle(v_opt))}')
##
##        v_est = bf_vector_est[i]
##        label_text.append(f'est (sem restricao)')
##        cb = np.column_stack((v_opt, v_est))
##        print (f'norm(v_est): {norm(v_est)}')
##        print (f'abs(v_est): \n{np.abs(v_est)}')
##        print (f'angle(v_est): \n{np.rad2deg(np.angle(v_est))}')
## 
##
##
##        for phase_resolution_bits in phase_resolution_bits_avaiable:
##            v_est_frab = bf_vector_est_frab_dict[phase_resolution_bits][i]
##            cb = np.column_stack((cb, v_est_frab))
##            label_text.append(f'est (EGT - {phase_resolution_bits} bits)')
##            print (f'norm(v_est_frab): {norm(v_est_frab)}')
##            print (f'abs(v_est_frab): \n{np.abs(v_est_frab)}')
##            print (f'angle(v_est_frab): \n{np.rad2deg(np.angle(v_est_frab))}')
##            #print (np.shape(bf_vector_est_frab_dict[phase_resolution_bits]))
##            pass
##
##        v_dft = bf_vector_dft[i]
##        label_text.append(f'dft (EGT)')
##        cb = np.column_stack((cb, v_dft))
##        print (f'norm(v_dft): {norm(v_dft)}')
##        print (f'abs(v_dft): \n{np.abs(v_dft)}')
##        print (f'angle(v_dft): \n{np.rad2deg(np.angle(v_dft))}')
## 
##
##        plot_beamforming_from_codeword(cb, label_text)
#
#
#
#
#
