import numpy as np
import sys
from scipy.constants import c
from numpy.linalg import svd, matrix_rank, norm
#from lib.vq.utils import norm
import matplotlib.pyplot as plt
from utils import richscatteringchnmtx, wf, siso_c, mimo_csir_c, mimo_csit_eigenbeamforming_c, mimo_csit_beamforming, get_orthonormal_basis
import json
from utils import decode_codebook, get_codebook, beamsweeping



if __name__ == "__main__":
   
    
    # getting samples of channel
    print("plot of max eigenvalues of channels...")
    
    #np.random.seed(1234)
    #channels = np.load('/home/snow/github/land/dataset/npy_files_s007/training_set_4x4.npy')
    #title = f'richscattering'

    #seed = np.array([795]) 
    seed = np.random.choice(1000, 1)
    print (f'seed: {seed}')
    np.random.seed(seed)

    #samples_pathfile = f'samples_random_4x4.npy'
    #samples_pathfile = f'mimo_fixed_v.npy'
    samples_pathfile = sys.argv[1]
    channels = np.load(samples_pathfile)
    print (f'shape of samples: {np.shape(channels)}')
    title = f'seed-{seed}-'+ samples_pathfile
    #channels = np.load('nlos_channels_4x4.npy')

    #snr_db = np.arange(-20, 20, 1)
    #snr = 10 ** (snr_db/10)

    #c_siso_all = []
    #c_mimo_csir_all = []
    #c_mimo_csit_eigenbeamforming_all = []
    #c_mimo_csit_beamforming_all = []

    ch_id_list = np.random.choice(len(channels), 100, replace=False)
    counter = 0

    result_pathfile = sys.argv[2]
    cb_est = get_codebook(result_pathfile)
    #for k, v in cb_est.items():
    #    print ('*********')
    #    print (k)
    #    print (v)
    #    v = np.matrix(v)
    #    print (f'norm of cw: {norm(v)}')
    #    print (np.shape(v))

    #eigenvalues = []
    gain_est_values = []
    gain_opt_values = []
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

        gain_est, cw_id_tx, cw_id_rx = beamsweeping(ch, cb_est)
        gain_est_values.append(gain_est)
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
    plt.plot(gain_est_values, '-.', label='gain_est')
    #plt.plot(s_est_max_values, '-', label='s_est_values')
    #plt.plot(np.array(eigenvalues) - np.array(s_est_values), '--')
    plt.title(f'{title} n, m = [{n, m}]')
    plt.xlabel(f'channel realization')
    plt.ylabel(f'max eigenvalue')
    plt.grid()
    plt.legend()
    plt.show()
    #plt.savefig(f'seed-{seed[0]}-channels-{n}x{m}-plot.png')
    print (seed)
