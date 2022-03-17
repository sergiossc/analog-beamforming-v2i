#!/usr/bin/python3.8
import numpy as np
from numpy.linalg import matrix_rank, norm, det, svd
#from numpy.linalg import matrix_rank
import load_lib
import lib.mimo.arrayconfig as phased_array
from utils import loschnmtx, richscatteringchnmtx, covariance_matrix, plot_errorbar
import matplotlib.pyplot as plt


#num_samples = 0
#s_los_list = []
#s_w_list = []
#s_ricean_k0_list = []
#s_ricean_k75_list = []
#s_ricean_k150_list = []
#s_ricean_k225_list = []



#------------------------------------
if __name__ == "__main__":
    rx_array = phased_array.rx_array
    tx_array = phased_array.tx_array

    # getting samples of channel
    print("Getting some samples of channel...")
    
    #np.random.seed(1234)
    #channels = np.load('/home/snow/github/land/dataset/npy_files_s007/training_set_4x4.npy')
    #title = f'richscattering'
    title = f'./npy_files/s002_rx5_nlos_channels_4x4.npy'
    channels = np.load(title)
    print (channels.shape)
    #channels = np.load('nlos_channels_4x4.npy')
    num_of_samples, num_of_cols, num_rows = channels.shape
    #num_of_samples = 10
    ch_id_list = np.random.choice(len(channels), num_of_samples, replace=False)
    counter = 0
 
    s_rt_list = []

    for ch_id in ch_id_list:
        counter = counter + 1
        #print (counter)
        ch = channels[ch_id]
        #print (ch.shape)

        h_rt = tx_array.size * ch/norm(ch)
        #print (h_rt.shape)
        u_rt, s_rt, vh_rt = svd(h_rt)
        s_rt_list.append(s_rt ** 2) # ** 2)
    plot_errorbar(np.array(s_rt_list), 's_rt', title)
    plt.show()














#----------------------------------------------------------------------










    #snr_db = np.arange(-20, 20, 0.01)
    #snr = 10 ** (snr_db/10)

    #c_siso_all = []
    #c_mimo_csir_all = []
    #c_mimo_csit_eigenbeamforming_all = []
    #c_mimo_csit_beamforming_all = []

    #ch_id_list = np.random.choice(len(channels), 100, replace=False)
    #counter = 0
 
#    for ch_id in ch_id_list:
#        ch = channels[ch_id]
#        #ch = richscatteringchnmtx(16, 16, 1.0)
#        counter += 1
#        print (f'{counter} of {len(ch_id_list)}')
#        n = np.shape(ch)[0]
#        m = np.shape(ch)[1]
#        ch = ch/norm(ch)
#        ch = m * ch
#    
#    
#        c_siso = []
#        c_mimo_csir = []
#        c_mimo_csit_eigenbeamforming = []
#        c_mimo_csit_beamforming = []
#    
#        #u, s, vh = svd(ch * ch.conj().T)    
#        u, s, vh = svd(ch * ch.conj().T)    # singular values 
#        s = s ** 2 #eigenvalues
#        #print (f's: {s}')
#        #print (matrix_rank(ch))
#        #print (np.sum(s))
#    
#        for snr_v in snr:
#            c_siso.append(siso_c(snr_v))
#            c_mimo_csir.append(mimo_csir_c(s, snr_v))
#            c_mimo_csit_eigenbeamforming.append(mimo_csit_eigenbeamforming_c(s, snr_v))
#            c_mimo_csit_beamforming.append(mimo_csit_beamforming(s, snr_v))
#
#        c_siso_all.append(c_siso)
#        c_mimo_csir_all.append(c_mimo_csir)
#        c_mimo_csit_eigenbeamforming_all.append(c_mimo_csit_eigenbeamforming)
#        c_mimo_csit_beamforming_all.append(c_mimo_csit_beamforming)
#
#
#
#
#
#
#
##---------------------------------------
#
#
#
