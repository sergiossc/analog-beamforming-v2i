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

    
    profile_pathfile_json = 'profile-rt-s002.json'
    result_pathfile_json = sys.argv[1] 

    #pathfile = result_pathfile_json

    with open(profile_pathfile_json) as profile:
        profile_data = profile.read()
        profile_d = json.loads(profile_data)

    test_channels_pathfile = profile_d['test_channel_samples_files'][0]
    print (test_channels_pathfile)


    with open(result_pathfile_json) as result:
        result_data = result.read()
        result_d = json.loads(result_data)



    channels = np.load(f'{test_channels_pathfile}')
    n_samples, n_rx, n_tx = np.shape(channels)
    print (np.shape(channels))

    # Getting estimated codebook from VQ training
    cb_dict = decode_codebook(result_d['codebook'])
    for k, v in cb_dict.items():
        cb_dict[k] = 1/np.sqrt(n_tx) * np.exp(1j * np.angle(v))

    # Getting a frab version of cb_dict
    cb_dict_frab = {}
    phase_resolution_bits_avaiable = [1, 2, 3, 4]
    #for k, v in cb_dict.items():
    for phase_resolution_bits in phase_resolution_bits_avaiable:
        cb_dict_frab[phase_resolution_bits] = {} #get_frab(v, phase_resolution_bits)
    #        #cb_dict_frab[k][phase_resolution_bits] = get_frab(v, phase_resolution_bits)
    #for k, v in cb_dict.items():
    for phase_resolution_bits in phase_resolution_bits_avaiable:
        #cb_dict_frab[k] = {} #get_frab(v, phase_resolution_bits)
        print (phase_resolution_bits)
        for k, v in cb_dict.items():
            #v_frab = get_frab(v, phase_resolution_bits)
            #cb_dict_frab[phase_resolution_bits][k] = v_frab
            cb_dict_frab[phase_resolution_bits][k] = 1/np.sqrt(n_tx) * np.exp(1j * np.angle(get_frab(v, phase_resolution_bits)))
            #print (cb_dict_frab[phase_resolution_bits][k])


    # Getting DFT codebook
    #fftmat, ifftmat = fftmatrix(n_tx, None)
    #cb_dft = matrix2dict(fftmat)
    #for k, v in cb_dft.items():
    #    cb_dft[k] = v.T
    #for k, v in cb_dft.items():
    #    print (np.shape(v))
    #    print (norm(v))


    num_of_trials = 50
    np.random.seed(5678)
    ch_id_list = np.random.choice(len(channels), num_of_trials, replace=False)
        
 
    bf_gain_opt = []
    bf_vector_opt = []
    bf_gain_est = []
    bf_vector_est = []

    bf_gain_est_frab_dict = {}
    bf_vector_est_frab_dict = {}
    for phase_resolution_bits in phase_resolution_bits_avaiable:
        bf_gain_est_frab_dict[phase_resolution_bits] = [] #et_frab(v, phase_resolution_bits)
        bf_vector_est_frab_dict[phase_resolution_bits] = [] #et_frab(v, phase_resolution_bits)

    #bf_gain_est_frab = []
    bf_gain_dft = []
    bf_vector_dft = []
#
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
        f = 1/np.sqrt(num_tx) * np.exp(1j * np.angle(f))
        gain_opt = norm(w.conj().T * (ch * f.conj())) ** 2
        #print (gain_opt)
        bf_vector_opt.append(f)
        bf_gain_opt.append(gain_opt)

        gain_est, cw_id_tx = beamsweeping2(ch, cb_dict)
        bf_vector_est.append(cb_dict[cw_id_tx])
        bf_gain_est.append(gain_est)
        #print (gain_est)

        #for k, v in cb_dict.items():
        for phase_resolution_bits in phase_resolution_bits_avaiable:
            gain_est_frab, cw_id_tx_frab = beamsweeping2(ch, cb_dict_frab[phase_resolution_bits])
            bf_gain_est_frab_dict[phase_resolution_bits].append(gain_est_frab)
            bf_vector_est_frab_dict[phase_resolution_bits].append(cb_dict_frab[phase_resolution_bits][cw_id_tx_frab])
            #print (gain_est_frab)
            pass

#        gain_dft, cw_id_tx_dft = beamsweeping2(ch, cb_dft)
#        bf_gain_dft.append(gain_dft)
#        bf_vector_dft.append(cb_dft[cw_id_tx_dft])
#        print (gain_dft)
#
    count_marker = 0
    markers_list = ["*", "h", "p", 4, 5, 6, 7, 8, 9, 10, 11, "+", "."]
    color_dict = {'opt': 'darkviolet', 'est': 'black', 'frab1': 'green', 'frab2': 'red', 'frab3': 'orange', 'frab4': 'blue' }
    label_dict = {'opt': 'ideal (EGT)', 'est': 'xiaoxiao (EGT)', 'frab1': 'xiaoxiao (EGT) - 1 bit', 'frab2': 'xiaoxiao (EGT) - 2 bit', 'frab3': 'xiaoxiao (EGT) - 3 bit', 'frab4': 'xiaoxiao (EGT) - 4 bit' }

    #plt.plot(bf_gain_opt, marker=markers_list[count_marker], label=label_dict['opt'], color=color_dict['opt'])
    plt.plot(bf_gain_opt, marker=markers_list[count_marker], label='ideal (EGT)', color=color_dict['opt'])
    count_marker += 1
    #plt.plot(bf_gain_est, marker=markers_list[count_marker], label=label_dict['est'], color=color_dict['est'])
    plt.plot(bf_gain_est, marker=markers_list[count_marker], label='xiaoxiao (EGT)', color=color_dict['est'])
    count_marker += 1
    #for k, v in cb_dict.items():

    for phase_resolution_bits in phase_resolution_bits_avaiable:
        plt.plot(bf_gain_est_frab_dict[phase_resolution_bits], marker=markers_list[count_marker], label=f'xiaoxiao (EGT) - {phase_resolution_bits} bit', color=color_dict[f'frab{phase_resolution_bits}'])
        count_marker += 1
    #plt.plot(bf_gain_dft, marker=markers_list[count_marker],  label=f'bf_gain_dft')
    plt.legend()
    plt.xlabel('Realizações de canal')
    plt.ylabel(r'Ganho de beamforming ($|w^{H}_{opt} \times (H \times f^{*}|^{2})$')
    #plt.title(f'{test_channels_pathfile}')
    fig_filename = f'random-s002-n4-bf-gain-resolution.png'
    plt.savefig(fig_filename, bbox_inches='tight')
    #plt.show()

    for i in range(num_of_trials):
        print (f'----------------------------------------------------------------------------x')
        label_text = []

        v_opt = bf_vector_opt[i]
        label_text.append(f'opt')

        v_est = bf_vector_est[i]
        label_text.append(f'est')
        cb = np.column_stack((v_opt, v_est))
 


        for phase_resolution_bits in phase_resolution_bits_avaiable:
            v_est_frab = bf_vector_est_frab_dict[phase_resolution_bits][i]
            cb = np.column_stack((cb, v_est_frab))
            label_text.append(f'frab{phase_resolution_bits}')

 
        color_list = [color_dict[k] for k in label_text]
        plot_beamforming_from_codeword(cb, label_text, color_list , label_dict, i)
        #plot_beamforming_from_codeword(v_opt, label_text, color_list , label_dict, i)
