#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sergiossc@gmail.com
"""
import numpy as np
from utils import *
from numpy.linalg import svd


if __name__ == '__main__':
    print ('Precoding on perfect CSI - MRT\n') 

    n_tx = 16
    n_rx = 1
    
    print (f'# of TX elements: {n_tx}')
    print (f'# of RX elements: {n_rx}')

    k = n_tx # codebook length. codeword number.
    
    num_of_elements = n_tx
    variance_of_samples = 1.0
    initial_alphabet_opt = 'katsavounidis'
    distortion_measure_opt = 'gain'
    num_of_samples = 1600
    max_num_of_interactions = 1000
    results_dir = '/home/snow/github/land/lloyd-gla/results'
    instance_id = str(uuid.uuid4())
    print (f'instance_id: {instance_id}')
    percentage_of_sub_samples = 1
    samples_random_seed = np.random.choice(10000)
    trial_random_seed = np.random.choice(10000)

    p = {'num_of_elements': num_of_elements, 'variance_of_samples': variance_of_samples, 'initial_alphabet_opt':initial_alphabet_opt, 'distortion_measure_opt':distortion_measure_opt, 'num_of_samples':num_of_samples, 'max_num_of_interactions':max_num_of_interactions, 'results_dir': results_dir, 'instance_id': instance_id, 'percentage_of_sub_samples': percentage_of_sub_samples, 'samples_random_seed': samples_random_seed, 'trial_random_seed': trial_random_seed}

    result = run_lloyd_gla(p)


#    print (f'cb:\n{cb}')
#    print (f'abs.cb:\n{np.abs(cb)}')
#    print (f'angle.cb:\n{np.rad2deg(np.angle(cb))}')
#    
#
#    print (f'samples.shape:\n{samples.shape}')
#    #print ('Now, using Kronecker product:\n')
#    gain_sum = 0
#    gain_sum_1 = 0
#    rate_sum = 0
#    rate_sum_1 = 0
#    for sample in samples:
#        print (np.shape(sample.T))
#        wr =  sample/norm(sample)
#        wr_1 = np.exp(1j * np.angle(wr))
#        wr_1 = wr_1/norm(wr_1)
#       
#        sample_normalized = sample/norm(sample)
#        print (f'wr: \n{wr}')
#        print (f'wr.abs: \n{np.abs(wr)}')
#        print (f'wr_1.abs: \n{np.abs(wr_1)}')
#        print (f'wr_1.angle(deg): \n{np.rad2deg(np.angle(wr_1))}')
#        print (f'wr.angle(deg): \n{np.rad2deg(np.angle(wr))}')
#        print (f'sample_normalized: \n{sample_normalized}')
#        print (f'norm(wr): {norm(wr)}')
#        print (f'norm(wr_1): {norm(wr_1)}')
#        print (norm(sample))
#        gain = np.abs(np.sum(sample * wr.conj())) ** 2
#        gain_1 = np.abs(np.sum(sample * wr_1.conj())) ** 2
#        rate = np.log2(1+gain)
#        rate_1 = np.log2(1+gain_1)
#        print (f'gain: {gain}')
#        print (f'gain_1: {gain_1}')
#        print (f'rate: : {rate}')
#        print (f'rate_1: : {rate_1}')
#        gain_sum_1 += gain_1
#        gain_sum += gain
#        rate_sum_1 += rate_1
#        rate_sum += rate
#
#    print (f'gain_mean: {gain_sum/num_of_samples}')
#    print (f'gain_mean_1: {gain_sum_1/num_of_samples}')
#    print (f'rate_mean: {rate_sum/num_of_samples}')
#    print (f'rate_mean_1: {rate_sum_1/num_of_samples}')
#
#    #count = 0
# 
#    #for cw in cb:
#    #    print ('++++')
#    #    x = np.array(cw)
#    #    y = np.array(cw)
#    #    
#    #    z = np.kron(x,y)
#    #    count += 1
#    #    print (f'shape of z: \n{np.size(z)}')
#    #    print (f'norm of z: \n{norm(z)}')
#    #    print (f'abs of z: \n{np.abs(z)}')
#    #    print (f'angle of z: \n{np.rad2deg(np.angle(z))}')
#        
#    #print (f'count: {count}')
#    #variance = 1.0
#    #seed = 12345
#
#    #dft_samples = gen_samples(dft_cb, l, variance, seed)
#    #non_dft_samples = gen_samples(None, l, variance, seed, n_rx, n_tx)
#
#    #samples = non_dft_samples
#
#    #initial_codebook = np.array([samples[i] for i in np.random.choice(len(samples), k, replace=False)])
#   
#    #for sample in samples:
#    #    pass
##        print ('+++++')
##        h_mat = np.array(h_vec).reshape(n_rx, n_tx)
##        h_mat_normalized = h_mat/norm(h_mat)
##        print (f'norm mat: {norm(h_mat)}')
##        print (f'norm mat normalized: {norm(h_mat_normalized)}')
##        s, d, vh = svd(h_mat)
##        print (f's.shape: {s.shape}')
##        print (f'd.shape: {d.shape}')
##        print (f'vh.shape: {vh.shape}')
##        f = vh.conj().T
##        for i in f:
##            i = np.array(i).reshape(1, n_tx)
##            #print (i.shape)
##            pass
##
##
