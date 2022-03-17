import os
import numpy as np
import sys
from scipy.constants import c
from numpy.linalg import svd, matrix_rank, norm
#from lib.vq.utils import norm
import matplotlib.pyplot as plt
from utils import fftmatrix
import json
from utils import decode_codebook, get_codebook, check_files, beamsweeping2, matrix2dict



if __name__ == "__main__":
   

    #profile_pathfile = 'profile-rt-s002.json' 
    profile_pathfile = str(sys.argv[1])
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
    n = int(sys.argv[2])

    rx_array_size = n
    tx_array_size = n

    initial_alphabet_opt = str(sys.argv[3])
    
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
            num_of_levels = d_result['num_of_levels']

            # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
            #num_of_levels = d['num_of_levels']
            cb_est = get_codebook(pathfile)
            #print (f'-----> {len(cb_est.keys())}')
            #for k, v in cb_est.items():
            #    print (np.shape(v))
            cb_est_dict[pathfile] = cb_est
            cb_est_num_of_levels_dict[pathfile] = num_of_levels
    print (count)
    print (len(cb_est_dict.keys()))
        

    
    # getting samples of channel
    print("plot of max eigenvalues of channels...")
    
    #np.random.seed(1234)
    #channels = np.load('/home/snow/github/land/dataset/npy_files_s007/training_set_4x4.npy')
    #title = f'richscattering'

    #seed = np.array([795]) 
    seed = np.random.choice(1000, 1)
    print (f'seed: {seed}')
    seed = 5678
    np.random.seed(seed)

    #samples_pathfile = f'samples_random_4x4.npy'
    #samples_pathfile = f'mimo_fixed_v.npy'
    #samples_pathfile = sys.argv[1]
    channels = np.load(samples_pathfile)
    print (f'shape of samples: {np.shape(channels)}')
    #title = f'seed-{seed}-'+ samples_pathfile
    title = samples_pathfile
    #channels = np.load('nlos_channels_4x4.npy')

    snr_db = np.arange(-20, 20, 10)
    snr = 10 ** (snr_db/10)

    #c_siso_all = []
    #c_mimo_csir_all = []
    #c_mimo_csit_eigenbeamforming_all = []
    #c_mimo_csit_beamforming_all = []

    ch_id_list = np.random.choice(len(channels), 5, replace=False)
    counter = 0

    #result_pathfile = sys.argv[2]
    #cb_est = get_codebook(result_pathfile)
    #for k, v in cb_est.items():
    #    print ('*********')
    #    print (k)
    #    print (v)
    #    v = np.matrix(v)
    #    print (f'norm of cw: {norm(v)}')
    #    print (np.shape(v))
    pass #HERE
    mean_capacity_est_dict = {}
    for cb_id, cb_est in cb_est_dict.items():
        mean_capacity_est_dict[cb_id] = []
    mean_capacity_opt = []
    mean_capacity_dft_cb = []
    
    fftmat, ifftmat = fftmatrix(n)
    dft_cb = matrix2dict(fftmat) # each col is a cw dft
    for snr_v in snr:
        pass
        #eigenvalues = []
        gain_opt_values = []
        capacity_opt_values = []
        capacity_dft_cb_values = []
        capacity_est_values_dict = {}
        gain_est_dict = {}

        for cb_id, cb_est in cb_est_dict.items():
            gain_est_dict[cb_id] = []
            capacity_est_values_dict[cb_id] = []
        #s_est_max_values = []
 
        for ch_id in ch_id_list:
            ch = np.matrix(channels[ch_id])
            counter += 1
            n = np.shape(ch)[0]
            m = np.shape(ch)[1]
            ch = ch/norm(ch)
            ch = np.sqrt(m*n) * ch
    
            u, s, vh = svd(ch)    
            #u, s, vh = svd(ch * ch.conj().T)    # singular values 
            f = vh[0,:]
            w = u[:,0]
            gain_opt = norm(w.conj().T * (ch * f.conj().T)) ** 2
            gain_opt_values.append(gain_opt)
            capacity_opt_value = np.log2(1 + snr_v * gain_opt)
            capacity_opt_values.append(capacity_opt_value)


            gain_dft, cw_dft_id_tx = beamsweeping2(ch, dft_cb)
            capacity_dft_cb_values.append(np.log2(1 + snr * gain_dft))

            for cb_id, cb_est in cb_est_dict.items():
                pass
                gain_est, cw_id_tx = beamsweeping2(ch, cb_est)
                gain_est_dict[cb_id].append(gain_est)

                capacity_est_value = np.log2(1 + snr_v * gain_est)
                capacity_est_values_dict[cb_id].append(capacity_est_value)
        #plt.ticklabel_format(useOffset=False)
        #plt.plot(gain_opt_values, '-.', label='gain_opt')

        gain_est_dict = dict(sorted(gain_est_dict.items()))
        cb_est_num_of_levels_dict = dict(sorted(cb_est_num_of_levels_dict.items()))

        for k, v in gain_est_dict.items():
            #plt.plot(v, '-.', label=f'L={cb_est_num_of_levels_dict[k]}')
            pass
        #plt.title(f'{title}')
        #plt.xlabel(f'realização do canal de testes')
        #plt.ylabel(f'max ganho')
        #plt.grid()
        #plt.legend()
        #plt.show()
        print (seed)
        print (f'snr_v: {snr_v}')
        print (f'capacity_opt_values: {capacity_opt_values}')
        mean_capacity_opt.append(np.mean(capacity_opt_values))
        mean_capacity_dft_cb.append(np.mean(capacity_dft_cb_values))
        for cb_id, capacity_est_value in capacity_est_values_dict.items():
            pass
            mean_capacity_est_dict[cb_id].append(np.mean(capacity_est_value))
    plt.plot(snr, mean_capacity_opt)
    plt.plot(snr, mean_capacity_dft_cb, label='dft cb')
    for k, v in mean_capacity_est_dict.items():
        plt.plot(snr, v)
    plt.show()
