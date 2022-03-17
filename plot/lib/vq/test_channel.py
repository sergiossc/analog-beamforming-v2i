from utils import *
import numpy as np
from numpy.linalg import svd

num_tx_list = [4, 8, 16, 32, 64, 128]

for num_tx in num_tx_list:
    #num_tx = 16
    num_rx = 1
    
    num_of_samples = 40000
    
    samples_seed = 12345
    
    samples = gen_samples(None, num_of_samples, 1.0, samples_seed, num_rx, num_tx) 
    
    oversampling_factor = 1
    dft_codebook = gen_dftcodebook(num_tx, oversampling_factor)
       
    dft_codebook_dict = matrix2dict(dft_codebook)
 
    channel_gain_sum = 0
    mrt_gain_sum = 0
    egt_gain_sum = 0
    dft_gain_sum = 0

    for s in samples:
        #print (f'\n')
        #s = np.array(s).reshape(num_rx, num_tx)
        ##print (f'sample shape: {np.shape(s)}\n')
        f_channel = s.conj() #.T/norm(s)
        f_mrt = s.conj()/norm(s)
        f_egt = (1/np.sqrt(num_tx)) * np.exp(1j * np.angle(f_channel))


        #print (f'f_equal.abs: {np.abs(f_equal)}') 
        #print (f'f_equal.angle: {np.angle(f_equal)}') 
        #print (f'f.abs: {np.abs(f)}') 
        #print (f'f.angle: {np.angle(f)}') 
        ##print (f'precoding: {norm(f)}\n')
        ##print (f'f:\n{f}')
        channel_gain = np.abs(np.sum(f_channel * s)) ** 2
        mrt_gain = np.abs(np.sum(f_mrt * s)) ** 2
        egt_gain = np.abs(np.sum(f_egt * s)) ** 2

        max_cw_id, dft_gain = gain_distortion(s, dft_codebook_dict)
 
        channel_gain_sum = channel_gain_sum + channel_gain
        mrt_gain_sum = mrt_gain_sum + mrt_gain
        egt_gain_sum = egt_gain_sum + egt_gain
        dft_gain_sum = dft_gain_sum + dft_gain
        ##print (f'gain: {gain}')
        #u, d, vh = svd(s.T)
        #print (f'd = np.sum(sqrt({np.sum(np.sqrt(d))}))\n') 
        #n = squared_norm(s)
        #print (f'norm: {n}')
        #print (f'*****')
    channel_gain_mean = channel_gain_sum/num_of_samples
    mrt_gain_mean = mrt_gain_sum/num_of_samples
    egt_gain_mean = egt_gain_sum/num_of_samples
    dft_gain_mean = dft_gain_sum/num_of_samples
    
    print (f'num_tx, channel_gain_meam, mrt_gain_mean, egt_gain_mean, dft_gain_mean: {num_tx, channel_gain_mean, mrt_gain_mean, egt_gain_mean, dft_gain_mean}')





