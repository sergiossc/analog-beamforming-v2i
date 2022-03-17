import os
import numpy as np
import sys
from scipy.constants import c
from numpy.linalg import svd, matrix_rank, norm
#from lib.vq.utils import norm
import matplotlib.pyplot as plt
import json
from utils import decode_codebook, get_codebook, beamsweeping, check_files, beamsweeping2, fftmatrix, matrix2dict, get_frab



if __name__ == "__main__":
   

    profile_pathfile = 'profile-random-4x4.json' 
    with open(profile_pathfile) as profile:
        data = profile.read()
        d = json.loads(data)

    prefix_pathfiles = d['results_directory']
    result_files = os.listdir(prefix_pathfiles)
    pathfiles = check_files(prefix_pathfiles, result_files)
    print ('# of json files: ', len(pathfiles))
    samples_pathfile = d['test_channel_samples_files'][0]

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
            num_of_levels = d_result['num_of_levels']

            # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
            #num_of_levels = d['num_of_levels']
            cb_est = get_codebook(pathfile)
            for k, v in cb_est.items():
                pass
                #v = get_frab(np.matrix(v), 2)
                #cb_est[k] = v/norm(v)
                print (f'norm of v ----> {norm(cb_est[k])}')
                #print (np.shape(np.matrix(v)))
         
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
    seed = 1234
    np.random.seed(seed)

    #samples_pathfile = f'samples_random_4x4.npy'
    #samples_pathfile = f'mimo_fixed_v.npy'
    #samples_pathfile = sys.argv[1]
    channels = np.load(samples_pathfile)
    print (f'shape of samples: {np.shape(channels)}')
    #title = f'seed-{seed}-'+ samples_pathfile
    title = samples_pathfile
    #channels = np.load('nlos_channels_4x4.npy')

    #snr_db = np.arange(-20, 20, 1)
    #snr = 10 ** (snr_db/10)

    #c_siso_all = []
    #c_mimo_csir_all = []
    #c_mimo_csit_eigenbeamforming_all = []
    #c_mimo_csit_beamforming_all = []

    ch_id_list = np.random.choice(len(channels), 50, replace=False)
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

    #eigenvalues = []
    gain_opt_values = []
    gain_dftcb_values = []
    fftmat, ifftmat = fftmatrix(n, None)
    dftcb = matrix2dict(fftmat.T)
    for k, v in dftcb.items():
        dftcb[k] = v.T

    gain_est_dict = {}
    for cb_id, cb_est in cb_est_dict.items():
        gain_est_dict[cb_id] = []
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


        gain_dftcb, cw_id_tx_dft = beamsweeping2(ch, dftcb)
        #print (f'------------------------> dft cw_id_tx.norm: {norm(dftcb[cw_id_tx_dft])}')
        gain_dftcb_values.append(gain_dftcb)

        for cb_id, cb_est in cb_est_dict.items():
            pass
            gain_est, cw_id_tx = beamsweeping2(ch, cb_est)
            #print (f'------------------------> cw_id_tx.norm: {norm(cb_est[cw_id_tx])}')
            gain_est_dict[cb_id].append(gain_est)
        #u, s, vh = svd(h_a * h_a.conj().T)    # singular values 
        #u, s, vh = svd(ch * ch.conj().T)    # singular values 
        #s = np.sqrt(s) #
        #s = s ** 2 #eigenvalues
        #eigenvalues.append(np.sum(s)) #[0])
        #eigenvalues.append(s[0] ** 2) #[0])
        #eigenvalues.append(s[0]) #[0])
        #print (s[0])
        #print (f's: {s}')
        #print (matrix_rank(ch))
        #print (np.sum(s ** 2))
    
        #for snr_v in snr:
        #    c_siso.append(siso_c(snr_v))
        #    c_mimo_csir.append(mimo_csir_c(s, snr_v))
        #    c_mimo_csit_eigenbeamforming.append(mimo_csit_eigenbeamforming_c(s, snr_v))
        #    c_mimo_csit_beamforming.append(mimo_csit_beamforming(s, snr_v))

        #c_siso_all.append(c_siso)
        #c_mimo_csir_all.append(c_mimo_csir)
        #c_mimo_csit_eigenbeamforming_all.append(c_mimo_csit_eigenbeamforming)
        #c_mimo_csit_beamforming_all.append(c_mimo_csit_beamforming)

    #mine = np.mean(c_mimo_csit_eigenbeamforming_all, axis=0)
    #print (np.shape(mine))
    #plt.plot(snr_db, np.mean(c_mimo_csit_eigenbeamforming_all, axis=0),'-', label=f'MIMO com CSIT (Eigenbeamforming)')
    #plt.plot(snr_db, np.mean(c_mimo_csir_all, axis=0),'--', label=f'MIMO com CSIR')
    #plt.plot(snr_db, np.mean(c_mimo_csit_beamforming_all, axis=0),'-.', label=f'MIMO com CSIT (Beamforming)')
    #plt.plot(snr_db, np.mean(c_siso_all, axis=0), ':', label=f'SISO')
    plt.ticklabel_format(useOffset=False)
    #plt.plot(eigenvalues, ':', label='eigenvalues')
    plt.plot(gain_opt_values, '-.', label='gain_opt')
    plt.plot(gain_dftcb_values, '-.', label='gain_dftcb')

    gain_est_dict = dict(sorted(gain_est_dict.items()))
    cb_est_num_of_levels_dict = dict(sorted(cb_est_num_of_levels_dict.items()))

    for k, v in gain_est_dict.items():
        #if cb_est_num_of_levels_dict[k] == 2:
        plt.plot(v, '-.', label=f'L={cb_est_num_of_levels_dict[k]}')
    #plt.plot(s_est_max_values, '-', label='s_est_values')
    #plt.plot(np.array(eigenvalues) - np.array(s_est_values), '--')
    plt.title(f'{title}')
    plt.xlabel(f'realização do canal de testes')
    plt.ylabel(f'max ganho')
    plt.grid()
    plt.legend()
    plt.show()
    #plt.savefig(f'seed-{seed[0]}-channels-{n}x{m}-plot.png')
    print (seed)
