import numpy as np
from numpy.linalg import svd, matrix_rank, norm
import matplotlib.pyplot as plt
from utils import * #richscatteringchnmtx, decode_codebook
import sys
import json
import pandas as pd
import os


#def beamsweeping(ch, cb_tx_dict, w):
#
#    gain_est_max = -np.Inf
#    cw_id_max_tx = ''
#    n_rx, n_tx = np.shape(ch)
#    
#    for k in cb_dict.keys():
#        cw_tx = np.matrix(cb_tx_dict[k]) # f -> 1 x Nt matrix
#        f_s = cw_tx
#        #gain_s = norm(ch * f_s.conj().T)
#
#        gain_s = np.dot(w, np.dot(ch, f_s.T))
#        gain_s = norm(gain_s[0,0]) ** 2 
#
#        if gain_s > gain_est_max:
#            gain_est_max = gain_s
#            cw_id_max_tx = k
#
#    return gain_est_max, cw_id_max_tx

if __name__ == "__main__":


    #profile_pathfile_json = 'profile-rt-s002.json'
    training_result_pathfile_json = sys.argv[1] 
    test_channels_pathfile = sys.argv[2] 
    print (test_channels_pathfile)

    with open(training_result_pathfile_json) as result:
        result_data = result.read()
        result_d = json.loads(result_data)

    channels = np.load(f'{test_channels_pathfile}')
    n_samples, n_rx, n_tx = np.shape(channels)
    print (np.shape(channels))

    # Getting estimated codebook from VQ training
    initial_alphabet_opt = result_d['initial_alphabet_opt']
    cb_dict_orig = decode_codebook(result_d['codebook'])
    cb_dict = {}
    cw_id = 0
    for k, v in cb_dict_orig.items():
        cw_id += 1
        cb_dict[f'cw{cw_id}'] = 1/np.sqrt(n_tx) * np.exp(1j * np.angle(v))

    # Getting a frab version of cb_dict
    phase_resolution_bits_avaiable = [1, 2, 3, 4]
    cb_dict_frab = {}
    for phase_resolution_bits in phase_resolution_bits_avaiable:
        cb_dict_frab[phase_resolution_bits] = {}

    for phase_resolution_bits in phase_resolution_bits_avaiable:
        for k, v in cb_dict.items():
            #cb_dict_frab[phase_resolution_bits][k] = 1/np.sqrt(n_tx) * np.exp(1j * np.angle(get_frab(v, phase_resolution_bits)))
            cb_dict_frab[phase_resolution_bits][k] = get_frab(v, phase_resolution_bits)
            print (norm(cb_dict_frab[phase_resolution_bits][k]))
            


    #for k, v in cb_dict.items():
    #    print (k)
    #    print (norm(v))
    #for k, v in cb_dict_frab.items():
    #    print (k)
    #    for k1, v1 in v.items(): 
    #        print (k1)
    #        print (norm(v1))
    #-----------------------------------------------------------

    

    #num_of_trials = 1000
    num_of_trials = n_samples,
    
    np.random.seed(5678)
    ch_id_list = np.random.choice(len(channels), num_of_trials, replace=False)
        
 
    bf_gain_opt = []
    bf_gain_est = []

    #bf_gain_est_frab_dict = {}
    #for phase_resolution_bits in phase_resolution_bits_avaiable:
    #    bf_gain_est_frab_dict[phase_resolution_bits] = [] #et_frab(v, phase_resolution_bits)



    snr_db = np.arange(-20, 20, 2)
    snr = 10 ** (snr_db/10)

    mean_capacity_bf_gain_opt = []
    mean_capacity_bf_gain_est = []
    mean_capacity_bf_gain_est_b_bits = {}
    for k in cb_dict_frab.keys():
        mean_capacity_bf_gain_est_b_bits[k] = []

    for snr_v in snr:
        capacity_bf_gain_opt = []
        capacity_bf_gain_est = []
        capacity_bf_gain_est_b_bits = {}
        for k in cb_dict_frab.keys():
            capacity_bf_gain_est_b_bits[k] = []

        print ('snr_v: ', snr_v)
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
            f = (1/np.sqrt(num_tx)) * np.exp(1j * np.angle(f))
            gain_opt = norm(w.conj().T * (ch * f.conj())) ** 2
            #print ('gain_opt: ', gain_opt)
            #bf_vector_opt.append(f)
            capacity_bf_gain_opt.append(np.log2(1 + snr_v * gain_opt))


            gain_est, cw_id_tx = beamsweeping2(ch, cb_dict)
            capacity_bf_gain_est.append(np.log2(1 + snr_v * gain_est))
            
            for k, v in cb_dict_frab.items():
                gain_est_b_bits, cw_id_tx = beamsweeping2(ch, v)
                capacity_bf_gain_est_b_bits[k].append(np.log2(1 + snr_v * gain_est_b_bits))


        for k, v in cb_dict_frab.items():
            pass
            mean_capacity_bf_gain_est_b_bits[k].append(np.mean(capacity_bf_gain_est_b_bits[k]))
        mean_capacity_bf_gain_opt.append(np.mean(capacity_bf_gain_opt))
        mean_capacity_bf_gain_est.append(np.mean(capacity_bf_gain_est))
    marker_dict = {1: 'h', 2: 'p', 3: 4, 4: 5}
    plt.plot(snr_db, mean_capacity_bf_gain_opt, marker="*", label='ideal (EGT)')        
    for k, v in mean_capacity_bf_gain_est_b_bits.items():
        print (f'----------------k: {k}')
        plt.plot(snr_db, v, marker=marker_dict[k], label=f'{initial_alphabet_opt} (EGT) - {k} bits')        
    plt.plot(snr_db, mean_capacity_bf_gain_est, marker=7, label=f'{initial_alphabet_opt} (EGT)')        
    plt.xlabel('SNR(dB)')
    plt.ylabel('Capacidade (bps/Hz)')
    plt.grid()
    plt.legend()
    #plt.show()
    fig_filename = f'capacity-s002-n{n_tx}-{initial_alphabet_opt}-phase-resolution.png'
    print (fig_filename)
    plt.savefig(fig_filename, bbox_inches='tight')
##    
##            gain_est, cw_id_tx = beamsweeping2(ch, cb_dict)
##            #bf_vector_est.append(cb_dict[cw_id_tx])
##            print ('gain_est: ', gain_est)
##            bf_gain_est.append(gain_est)
##    
##            #for k, v in cb_dict.items():
##            for phase_resolution_bits in phase_resolution_bits_avaiable:
##                gain_est_frab, cw_id_tx_frab = beamsweeping2(ch, cb_dict_frab[phase_resolution_bits])
##                bf_gain_est_frab_dict[phase_resolution_bits].append(gain_est_frab)
##                #bf_vector_est_frab_dict[phase_resolution_bits].append(cb_dict_frab[phase_resolution_bits][cw_id_tx_frab])
##                print ('phase_resolution_bits: ', gain_est_frab)
##                pass
##
####
###    count_marker = 0
###    markers_list = ["*", "h", "p", 4, 5, 6, 7, 8, 9, 10, 11, "+", "."]
###    color_dict = {'opt': 'darkviolet', 'est': 'black', 'frab1': 'green', 'frab2': 'red', 'frab3': 'orange', 'frab4': 'blue' }
###    label_dict = {'opt': 'BF ideal', 'est': 'random (EGT)', 'frab1': '1', 'frab2': '2', 'frab3': '3', 'frab4': '4' }
###
###    plt.plot(bf_gain_opt, marker=markers_list[count_marker], label=label_dict['opt'], color=color_dict['opt'])
###    count_marker += 1
###    plt.plot(bf_gain_est, marker=markers_list[count_marker], label=label_dict['est'], color=color_dict['est'])
###    count_marker += 1
###    #for k, v in cb_dict.items():
###
###    for phase_resolution_bits in phase_resolution_bits_avaiable:
###        plt.plot(bf_gain_est_frab_dict[phase_resolution_bits], marker=markers_list[count_marker], label=f'random (EGT) - {phase_resolution_bits} bit', color=color_dict[f'frab{phase_resolution_bits}'])
###        count_marker += 1
###    #plt.plot(bf_gain_dft, marker=markers_list[count_marker],  label=f'bf_gain_dft')
###    plt.legend()
###    plt.xlabel('Realizações de canal')
###    plt.ylabel(r'Ganho de beamforming ($|w^{H}_{opt} \times (H \times f^{*}|^{2})$')
###    #plt.title(f'{test_channels_pathfile}')
###    fig_filename = f'random-s002-n4-bf-gain-resolution.png'
###    #plt.savefig(fig_filename, bbox_inches='tight')
###    plt.show()
###
