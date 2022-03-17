import numpy as np
import sys
from scipy.constants import c
from numpy.linalg import svd, matrix_rank
from lib.vq.utils import norm
import matplotlib.pyplot as plt
from utils import richscatteringchnmtx, wf, siso_c, mimo_csir_c, mimo_csit_eigenbeamforming_c, mimo_csit_beamforming, get_orthonormal_basis

if __name__ == "__main__":
   
    
    # getting samples of channel
    print("Getting some samples of channel...")
    
    #np.random.seed(1234)
    #channels = np.load('/home/snow/github/land/dataset/npy_files_s007/training_set_4x4.npy')
    #channels = np.load('test_waterfiling.py')
    #title = f'richscattering'

    #seed = np.array([795]) 
    seed = np.random.choice(9000, 1)
    print (f'seed: {seed}')
    np.random.seed(seed)

    #samples_pathfile = f'samples_random_4x4.npy'
    #samples_pathfile = f'mimo_fixed_v.npy'
    samples_pathfile = sys.argv[1]
    channels = np.load(samples_pathfile)
    title = f'seed-{seed}-'+ samples_pathfile
    #channels = np.load('nlos_channels_4x4.npy')

    snr_db = np.arange(-20, 20, 1)
    snr = 10 ** (snr_db/10)

    c_siso_all = []
    c_mimo_csir_all = []
    c_mimo_csit_eigenbeamforming_all = []
    c_mimo_csit_beamforming_all = []

    ch_id_list = np.random.choice(len(channels), 100, replace=False)
    counter = 0
 
    for ch_id in ch_id_list:
        ch = np.matrix(channels[ch_id])
        #ch = richscatteringchnmtx(16, 16, 1.0)
        counter += 1
        print (f'{counter} of {len(ch_id_list)}')
        n = np.shape(ch)[0]
        m = np.shape(ch)[1]
        ch = ch/norm(ch)
        ch = m * ch
    
        fc = 60 * (10 ** 9)
        wavelength = c/fc
        d = 1/2
        nr = n
        nt = m
        #s_rx = get_orthonormal_basis(nr, d, wavelength)
        #s_tx = get_orthonormal_basis(nt, d, wavelength)
        #h_a = s_rx.conj().T * (ch * s_tx.conj())
        #h_a = nt * h_a/norm(h_a)
        #print (f'norm(h_a): {norm(h_a)}')
        #print (f'norm(ch): {norm(ch)}')
 
    
        c_siso = []
        c_mimo_csir = []
        c_mimo_csit_eigenbeamforming = []
        c_mimo_csit_beamforming = []
    
        #u, s, vh = svd(ch * ch.conj().T)    
        u, s, vh = svd(ch)    # vector s is the vector of singular values 
        """
            quando calculamos o svd (h h*), o s obtido ja representa os autovalores. 
            quando calculamos o svd(h), o s obtido precisa ser elevado ao quadrado para se obter os autovalores.
        """
        ##u, s, vh = svd(h_a * h_a.conj().T)    # singular values 
        #u, s, vh = svd(ch * ch.conj().T)    # singular values 
        s = s ** 2 #eigenvalues
        #s = np.sqrt(s) # ** 2 #eigenvalues
        #print (f's: {s}')
        #print (matrix_rank(ch))
        #print (np.sum(s))
    
        for snr_v in snr:
            c_siso.append(siso_c(snr_v))
            c_mimo_csir.append(mimo_csir_c(s, snr_v))
            c_mimo_csit_eigenbeamforming.append(mimo_csit_eigenbeamforming_c(s, snr_v))
            c_mimo_csit_beamforming.append(mimo_csit_beamforming(s, snr_v))

        c_siso_all.append(c_siso)
        c_mimo_csir_all.append(c_mimo_csir)
        c_mimo_csit_eigenbeamforming_all.append(c_mimo_csit_eigenbeamforming)
        c_mimo_csit_beamforming_all.append(c_mimo_csit_beamforming)

    #mine = np.mean(c_mimo_csit_eigenbeamforming_all, axis=0)
    #print (np.shape(mine))
    plt.plot(snr_db, np.mean(c_mimo_csit_eigenbeamforming_all, axis=0),'-', label=f'MIMO com CSIT (Eigen Beamforming)')
    #plt.plot(snr_db, np.mean(c_mimo_csir_all, axis=0),'--', label=f'MIMO com CSIR')
    plt.plot(snr_db, np.mean(c_mimo_csit_beamforming_all, axis=0),'-.', label=f'MIMO com CSIT (Beamforming)')
    #plt.plot(snr_db, np.mean(c_siso_all, axis=0), ':', label=f'SISO')
    #plt.title(f'{title} n, m = [{n, m}]')
    plt.xlabel(f'SNR (dB)')
    plt.ylabel(f'Capacidade (bps/Hz)')
    plt.grid()
    plt.legend()
    #plt.show()
    fig_filename = f'capacity-s002-channels-{n}x{m}-plot.png'
    plt.savefig(fig_filename, bbox_inches='tight')
    print (fig_filename)
