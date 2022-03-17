#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sergiossc@gmail.com
"""
import numpy as np
import uuid
import matplotlib.pyplot as plt
import json
from scipy.linalg import hadamard
import os

def squared_norm(cw):
    """
    Input: cw as a vector (1-dim)
    Output: return a squared norm as a inner product of cw.conj() * cw
    """
    f_cw = cw.flatten()
    inner_product = np.sum(f_cw.conj() * f_cw) 
    return inner_product

def norm(cw): 
    return np.sqrt(squared_norm(cw))


def gen_dftcodebook(num_of_cw, oversampling_factor=None):
    if oversampling_factor is not None:
        tx_array = np.arange(num_of_cw * int(oversampling_factor))
        mat = np.matrix(tx_array).T * tx_array
        cb = (1.0/np.sqrt(num_of_cw)) * np.exp(1j * 2 * np.pi * mat/(oversampling_factor * num_of_cw))
    elif oversampling_factor is None:
        tx_array = np.arange(num_of_cw)
        mat = np.matrix(tx_array).T * tx_array
        cb =  (1.0/np.sqrt(num_of_cw)) * np.exp(1j * 2 * np.pi * mat/num_of_cw)
    else:
        raise(f'Please chose \'None\' or int value for oversampling_factor')

    return cb[:, 0:num_of_cw]
    
def richscatteringchnmtx(num_rx, num_tx, variance):
    """
    Ergodic channel. Fast, frequence non-selective channel: y_n = H_n x_n + z_n.  
    Narrowband, MIMO channel
    PDF model: Rich Scattering
    Circurly Simmetric Complex Gaussian from: 
         https://www.researchgate.net/post/How_can_I_generate_circularly_symmetric_complex_gaussian_CSCG_noise
    """
    sigma = variance
    #my_seed = 2323
    #np.random.seed(my_seed)
    h = np.sqrt(sigma/2)*(np.random.randn(num_rx, num_tx) + np.random.randn(num_rx, num_tx) * 1j)
    #h = np.sqrt(sigma/2)*np.random.randn(num_tx, num_rx)
    return h

def gen_samples(codebook, num_of_samples, variance, seed, nrows = None, ncols = None):

    np.random.seed(seed)
    samples = []

    if codebook is not None:
        nrows = np.shape(codebook)[0]
        ncols = np.shape(codebook)[1]
        for n in range(int(num_of_samples/nrows)):
            for cw in codebook:
                noise = np.sqrt(variance/(2*ncols)) * (np.random.randn(1, ncols) + np.random.randn(1, ncols) * 1j)
                sample = cw + noise
                samples.append(sample)

    elif codebook is None:
        if (nrows and ncols) is not None:
            cw = np.zeros((1, nrows * ncols), dtype=complex)
            for n in range(num_of_samples):
                noise = np.sqrt(variance/(2*ncols*nrows)) * (np.random.randn(1, nrows * ncols) + np.random.randn(1, nrows * ncols) * 1j)
                sample = cw + noise
                samples.append(sample)
        else:
            #print ('Please, you shold give information about number of rows and cols of samples.')
            raise ValueError("Please, you shold give information about number of 'rows' and 'cols' of samples.")

    np.random.seed(None)

    return np.array(samples)

def covariance_matrix(samples):
    """
      https://handwiki.org/wiki/Complex_random_vector
    """
    mean = complex_average(samples)
    de_meaned = np.array([sample - mean for sample in samples])
    num_samples, num_rows, num_cols = de_meaned.shape
    S = np.zeros((num_cols, num_cols), dtype=complex)
    for col1 in range(num_cols):
        for col2 in range(num_cols):
            x = np.sum(de_meaned[:,:,col1].conj() * de_meaned[:,:,col2])/(num_samples-1)
            S[col1, col2] = x
            #if col1 == col2:
            #    pass
            #    #print (np.power(x, 2))
    #print ("S:\n")
    #for s in S:
        #print (f's: {s}\n')
    #print ("trace(S):\n", np.trace(S))
    return S

def complex_average(samples):
    return np.mean(samples, axis=0)

def duplicate_codebook(codebook, perturbation_vector):
    new_codebook = []
    for cw in codebook:
        cw1 = cw + perturbation_vector
        cw2 = cw - perturbation_vector
        cw1 = cw1/norm(cw1)
        cw2 = cw2/norm(cw2)
        new_codebook.append(cw1)
        new_codebook.append(cw2)
    return np.array(new_codebook)

def dict2matrix(dict_info):
    vector = []
    for k, v in dict_info.items():
        vector.append(v)
    return np.array(vector)

def matrix2dict(matrix):
    dict_info = {}
    for l in matrix:
        id = uuid.uuid4()
        dict_info[id] = l
    return dict_info

def sorted_samples(samples, attr='norm'):
    nsamples, nrows, ncols = samples.shape
    s_not_sorted = []

    if attr == 'norm': #Sorted by vector norm   ??????
        for s in samples:
            s_norm = np.abs(norm(s))
            s_info = {}
            s_info = {'s_norm': s_norm, 's': s}
            s_not_sorted.append(s_info)

        s_sorted = sorted(s_not_sorted, key=lambda k: k['s_norm'])
        samples_sorted = [v['s'] for v in s_sorted]
        attr_sorted = [v['s_norm'] for v in s_sorted]

    elif attr == 'mse': #Sorted by vector norm   ??????
        s_avg = complex_average(samples)
        for s in samples:
            s_mse = norm(s-s_avg)
            s_info = {}
            s_info = {'s_mse': s_mse, 's': s}
            s_not_sorted.append(s_info)

        s_sorted = sorted(s_not_sorted, key=lambda k: k['s_mse'])
        samples_sorted = [v['s'] for v in s_sorted]
        attr_sorted = [v['s_mse'] for v in s_sorted]

    elif attr == 'stddev':  #Sorted by Standard Deviation

        s_avg = complex_average(samples)
        for s in samples:
            s_de_meaned = s - s_avg
            s_stddev = squared_norm(s_de_meaned)/ncols
            s_info = {}
            s_info = {'s_stddev': s_stddev, 's': s}
            s_not_sorted.append(s_info)

        s_sorted = sorted(s_not_sorted, key=lambda k: k['s_stddev'])
        samples_sorted = [v['s'] for v in s_sorted]
        attr_sorted = [v['s_stddev'] for v in s_sorted]
        var = sum(attr_sorted)/len(attr_sorted)
        std = np.sqrt(var)
        print ("var: ", var)
        print ("std: ", std)

    elif attr == 'abs_mean_characteristic_value': # From the paper

        for s in samples:
            num_rx, num_tx = s.shape
            #print (num_rx, num_tx)
            #print ("s:\n", s)
            s_mean = np.sum(s)/num_tx
            #print ("s_avg:\n", s_avg)
            s_info = {}
            s_info = {'s_abs_mean': np.abs(s_mean), 's': s}
            s_not_sorted.append(s_info)

        s_sorted = np.array(sorted(s_not_sorted, key=lambda k: k['s_abs_mean']))
        samples_sorted = [v['s'] for v in s_sorted]
        attr_sorted = [v['s_abs_mean'] for v in s_sorted]

    elif attr == 'variance_characteristic_value': # From the paper

        #s_avg = complex_average(samples)
        for s in samples:
            num_rx, num_tx = s.shape
            s_avg = np.sum(s)/num_tx
            s_demeaned = s - s_avg
            s_var = np.sqrt(np.sum(s_demeaned.conj() * s_demeaned)/num_tx)
            s_info = {}
            s_info = {'s_var': np.abs(s_var), 's': s}
            s_not_sorted.append(s_info)

        s_sorted = np.array(sorted(s_not_sorted, key=lambda k: k['s_var']))
        samples_sorted = [v['s'] for v in s_sorted]
        attr_sorted = [v['s_var'] for v in s_sorted]


    else:
        return -1

    return np.array(samples_sorted), np.array(attr_sorted)


def mse_distortion(sample, codebook_dict):
    min_mse = np.Inf
    min_cw_id = None
    for cw_id, cw in codebook_dict.items():
        mse = squared_norm(cw - sample)/np.size(sample)
        if mse < min_mse:
            min_mse = mse
            min_cw_id = cw_id
    return min_cw_id, min_mse

def gain_distortion(sample, codebook_dict):
    max_gain = -np.Inf
    max_cw_id = None
    len_sample = sample.shape[0] * sample.shape[1]
    sample = np.array(sample).reshape(len_sample)
    for cw_id, cw in codebook_dict.items():
        cw = np.array(cw).reshape(len_sample)
        #gain = np.abs(np.sum(cw * sample)) ** 2
        #sample = sample/norm(sample)
        #sample = sample/norm(sample)
        gain = np.cos(np.angle(np.sum(cw * sample.conj())))
        if gain > max_gain:
            max_gain = gain
            max_cw_id = cw_id
    return max_cw_id, max_gain

def gain_codeword_derivation(codeword, samples):
    #new_codeword = np.array([samples[i] for i in np.random.choice(len(samples), 1, replace=False)])
    theta_sum = 0.0
    theta_mean = 0.0

    len_codeword = codeword.shape[0] * codeword.shape[1]
    codeword = np.array(codeword).reshape(len_codeword)

    t_sum = np.zeros(len_codeword)

    for sample in samples:
        #theta = np.angle(2 * np.conjugate(np.inner(sample.conj(), sample)) * codeword)

        len_sample = sample.shape[0] * sample.shape[1]
        sample = np.array(sample).reshape(len_sample)

        h2 = [sample[i].conj()*sample[i] for i in range(len_sample)]
        h2 = np.array(h2)

        t = [codeword[i]*h2[i] for i in range(len_sample)]
        t = 2 * np.array(t)
        t = np.angle(t)
        
        #print (f't.shape: {t.shape}')
        #print (f'h2.shape: {h2.shape}')
        #print (f'sample.shape: {sample.shape}')
        #print (f'codeword.shape: {codeword.shape}')
        
        #theta_sum = theta_sum + theta
        t_sum = t_sum + t
    #theta_mean = theta_sum/len(samples)
    t_mean = t_sum/len(samples)
    #print (f't_mean.shape: {t_mean.shape}')
    step_size = 0.001
    #new_theta = np.angle(codeword) + step_size * theta_mean
    new_t = np.angle(codeword) + step_size * t_mean
    #new_codeword = np.exp(1j * new_theta)
    new_cw = np.exp(1j * new_t)
    #new_codeword = new_codeword/norm(new_codeword)
    new_cw = new_cw/norm(new_cw)
    new_cw = np.array(new_cw).reshape(1, len_codeword)
    #print (f'new_cw.shape: {new_cw.shape}')
    #return new_codeword
    return new_cw

def xiaoxiao_initial_codebook(samples):

    num_samples, num_rows, num_cols = samples.shape

    # Code samples in hadamard code
    samples_hadamard = hadamard_transform(samples, False)    

    # Ordering samples by variance characteristic value (ascending way)
    samples_sorted, attr_sorted = sorted_samples(samples_hadamard, 'variance_characteristic_value') 
    
    # Index A, B and C groups
    a_group_begin = 0
    a_group_end = 17 * int(num_samples/20)

    b_group_begin = a_group_end
    b_group_end = b_group_begin + (2 * int(num_samples/20))

    c_group_begin = b_group_end
    c_group_end = -1 

    # Getting samples from ordered samples spliting in groups as indexed as before
    a_group_of_samples = samples_sorted[a_group_begin:a_group_end, :, :]
    b_group_of_samples = samples_sorted[b_group_begin:b_group_end, :, :]
    c_group_of_samples = samples_sorted[c_group_begin:c_group_end, :, :]
    
    # Ordering subgroups by mean characteristic value
    samples_a_group_sorted, attr_a_group_sorted = sorted_samples(a_group_of_samples, 'abs_mean_characteristic_value') 
    samples_b_group_sorted, attr_a_group_sorted = sorted_samples(b_group_of_samples, 'abs_mean_characteristic_value') 
    samples_c_group_sorted, attr_a_group_sorted = sorted_samples(c_group_of_samples, 'abs_mean_characteristic_value') 

    # For each subgroup, select the codewords. Ex.: all/2, all/4 and all/4 number of codewords
    num_of_codewords = num_cols

    #print ('len(group_a): ', len(samples_a_group_sorted))
    index_a = get_index_codewords_from_sub_samples(len(samples_a_group_sorted), num_of_codewords/2)
    #print ('index_a:', index_a)

    #print ('len(group_b): ', len(samples_b_group_sorted))
    index_b = get_index_codewords_from_sub_samples(len(samples_b_group_sorted), num_of_codewords/4)
    #print ('index_b:', index_b)

    #print ('len(group_c): ', len(samples_c_group_sorted))
    index_c = get_index_codewords_from_sub_samples(len(samples_c_group_sorted), num_of_codewords/4)
    #print ('index_c:', index_c)


    #igetting codewords from subgroups
    list_initial_codebook_from_a_group = [samples_a_group_sorted[i]/norm(samples_a_group_sorted[i]) for i in index_a]
    list_initial_codebook_from_b_group = [samples_b_group_sorted[i]/norm(samples_b_group_sorted[i]) for i in index_b]
    list_initial_codebook_from_c_group = [samples_c_group_sorted[i]/norm(samples_c_group_sorted[i]) for i in index_c]

    initial_codebook = np.array(list_initial_codebook_from_a_group + list_initial_codebook_from_b_group + list_initial_codebook_from_c_group)

    #print (initial_codebook.shape)
    return initial_codebook, samples_hadamard

def get_index_codewords_from_sub_samples(n_samples, n_codewords):

    slot = int(n_samples/n_codewords)
    step = slot/2

    index_codebook_list = []

    for n in range(int(n_codewords)):
            start = n * slot
            mid = start + step
            index_codebook_list.append(int(mid))
    return index_codebook_list



def katsavounidis_initial_codebook(samples):

    num_samples, num_rows, num_cols = samples.shape


    samples_dict = matrix2dict(samples)
        
    max_norm = -np.Inf
    max_sample_id = ''

    for s_id, s in samples_dict.items():
        s_norm = norm(s)
        if s_norm > max_norm:
            max_norm = s_norm
            max_sample_id = s_id
    
    num_of_codewords = num_cols
    initial_codebook = np.zeros((num_of_codewords, num_rows, num_cols), dtype=complex)
    
    # Remove the max_sample_id from samples_dict and add it as our first codeword in initial_codebook
    initial_codebook[0,:,:] = samples_dict.pop(max_sample_id) 

    # Step 2: Define 2nd codeword as the largest distance from the 1st codeword
    cw = initial_codebook[0,:,:]
    max_distance = -np.Inf
    max_distance_sample_id = '' 
    for s_id, s in samples_dict.items():
        s_distance = norm(s - cw)
        if s_distance > max_distance:
            max_distance = s_distance
            max_distance_sample_id = s_id
    initial_codebook[1,:,:] = samples_dict.pop(max_distance_sample_id)

    # Step 3: defining next codewords

    for i in range(0, num_of_codewords - 2):

        min_distance = np.Inf
        min_distance_sample_id = '' 

        for s_id, s in samples_dict.items():
            s_distance = 0
            for codeword in initial_codebook:
                s_distance = s_distance + norm(s - codeword)
            if s_distance < min_distance:
                min_distance = s_distance
                min_distance_sample_id = s_id
    
        #for s_id, s in samples_dict.items():
        #    for codeword in initial_codebook:
        #        s_distance = norm(s - codeword)
        #        if s_distance < min_distance:
        #            min_distance = s_distance
        #            min_distance_sample_id = s_id
    
        max_distance = -np.Inf
        max_distance_sample_id = '' 

        for s_id, s in samples_dict.items():
    
            s_distance = norm(s - samples_dict[min_distance_sample_id])
            if s_distance > max_distance:
                max_distance = s_distance
                max_distance_sample_id = s_id
    
    
        initial_codebook[i+2,:,:] = samples_dict.pop(max_distance_sample_id)

    initial_codebook_normalized = np.zeros((num_of_codewords, num_rows, num_cols), dtype=complex)
    for i in range(num_of_codewords):
        initial_codebook_normalized[i,:,:] = initial_codebook[i,:,:]/norm(initial_codebook[i,:,:]) 

    return initial_codebook_normalized
 

def perform_distortion(sample, codebook_dict, metric):
    cw_id = None
    distortion = None
    distortion_opts = {'mse': mse_distortion, 'gain': gain_distortion}
    distortion_function = distortion_opts.get(metric, None)
    cw_id, distortion = distortion_function(sample, codebook_dict)
    return cw_id, distortion

def sa(initial_codebook, variance_of_samples, initial_temperature, max_iteractions, lloyd_num_of_interactions, distortion_measure_opt, num_of_levels, samples):
    
    best_lloydcodebook, best_sets, best_mean_distortion_by_round = lloyd_gla("sa", samples, num_of_levels, lloyd_num_of_interactions, distortion_measure_opt, initial_codebook)
    best_mean_distortion_list = list(best_mean_distortion_by_round[1])
    best_distortion = best_mean_distortion_list[-1]
    current_temperature = initial_temperature
    current_iteraction = 0
    while current_temperature > 0.01:
        print (current_temperature)
        while current_iteraction < max_iteractions:
            
            #candidate_codebook = gen_samples(initial_codebook, num_of_levels, variance_of_samples, None)
            candidate_codebook = np.array([samples[i] for i in np.random.choice(len(samples), num_of_levels, replace=False)])
            candidate_lloydcodebook, candidate_sets, candidate_mean_distortion_by_round = lloyd_gla("sa", samples, num_of_levels, lloyd_num_of_interactions, distortion_measure_opt, candidate_codebook)
            candidate_distortion_by_lloyd_interactions = list(candidate_mean_distortion_by_round[1])
            candidate_distortion = candidate_distortion_by_lloyd_interactions[-1]


            initial_lloydcodebook, initial_sets, initial_mean_distortion_by_round = lloyd_gla("sa", samples, num_of_levels, lloyd_num_of_interactions, distortion_measure_opt, initial_codebook)
            initial_distortion_by_lloyd_interactions = list(initial_mean_distortion_by_round[1])
            initial_distortion = initial_distortion_by_lloyd_interactions[-1]

            delta_distortion = candidate_distortion - initial_distortion
            if delta_distortion < 0:
                initial_codebook = candidate_codebook
                if (candidate_distortion < best_distortion):
                    best_distortion = candidate_distortion
                    best_lloydcodebook = candidate_lloydcodebook
                    best_sets = candidate_sets
                    best_mean_distortion_by_round = candidate_mean_distortion_by_round
                    #print ('candidate: ', candidate_distortion)
                    #print ('initial: ', initial_distortion)
            else:
                x = np.random.rand()
                if (x < np.exp(-delta_distortion/current_temperature)):
                    initial_codebook = candidate_codebook

            current_iteraction += 1
        current_temperature = current_temperature * 0.1
        current_iteraction = 0
    print (best_distortion)
    return best_lloydcodebook, best_sets, best_mean_distortion_by_round

def run_lloyd_gla(parm):

    data = {}

    trial_id = parm['trial_id']
    data['trial_id'] = trial_id

    results_dir = parm['results_dir']
    data['results_dir'] = results_dir

    #json_filename = str(results_dir) + '/' + str(instance_id) + '.json'
    json_filename = f'result_{trial_id}.json'

    initial_alphabet_opt = parm['initial_alphabet_opt']
    data['initial_alphabet_opt'] = initial_alphabet_opt

    distortion_measure_opt = parm['distortion_measure_opt']
    data['distortion_measure_opt'] = distortion_measure_opt

    max_num_of_interactions = parm['max_num_of_interactions']
    data['max_num_of_interactions'] = max_num_of_interactions

    trial_random_seed = parm['trial_random_seed']
    data['trial_random_seed'] = float(trial_random_seed)

    samples = parm['channel_samples']
    #samples = samples[0:100]
    num_of_samples, rx_array_size, tx_array_size = np.shape(samples)
    data['rx_array_size'] = rx_array_size
    data['tx_array_size'] = tx_array_size
    data['num_of_samples'] = num_of_samples
    num_of_levels = tx_array_size
    data['num_of_levels'] = num_of_levels

    # Starting lloyd with an specific initial alphabet opt
    if initial_alphabet_opt == 'xiaoxiao':
        initial_codebook, samples_hadamard = xiaoxiao_initial_codebook(samples)
        codebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)

    elif initial_alphabet_opt == 'katsavounidis':
        initial_codebook = katsavounidis_initial_codebook(samples)
        codebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)

    elif initial_alphabet_opt == 'sa':
        initial_codebook = np.array([samples[i] for i in np.random.choice(len(samples), num_of_levels, replace=False)])
        initial_temperature = 10
        sa_max_num_of_iteractions = 20
        variance_of_samples = 1.0 ######
        codebook, sets, mean_distortion_by_round = sa(initial_codebook, variance_of_samples, initial_temperature, sa_max_num_of_iteractions, max_num_of_interactions, distortion_measure_opt, num_of_levels, samples)

    elif initial_alphabet_opt == 'unitary_until_num_of_elements':
        initial_codebook = complex_average(samples)
        initial_codebook = initial_codebook/norm(initial_codebook)
        codebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)

    elif initial_alphabet_opt == 'random_from_samples':
        initial_codebook = np.array([samples[i] for i in np.random.choice(len(samples), num_of_levels, replace=False)])
        codebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)
 
    elif initial_alphabet_opt == 'random':
        angle_range = np.linspace(0, 2*np.pi, 360)
        initial_codebook = num_of_levels/np.sqrt(rx_array_size * tx_array_size) *  np.exp(1j * np.random.choice(angle_range, (num_of_levels, rx_array_size, tx_array_size), replace=True))
        codebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)
        

    #data['initial_codebook'] = encode_codebook(matrix2dict(initial_codebook))
    #data['codebook'] = encode_codebook(matrix2dict(codebook))
    data['codebook'] = encode_codebook(codebook)
    ##plot_performance(mean_distortion_by_round, 'MSE as distortion', 'distortion.png')
    #codebook = 1/np.sqrt(num_of_elements) * np.exp(1j * np.angle(codebook))
    #data['egt_codebook'] = encode_codebook(matrix2dict(lloydcodebook))
    data['sets'] = encode_sets(sets)
    data['mean_distortion_by_round'] = encode_mean_distortion(mean_distortion_by_round)

    with open(json_filename, "w") as write_file:
        json.dump(data, write_file, indent=4)

    return 0

def lloyd_gla(initial_alphabet_opt, samples, num_of_levels, num_of_iteractions, distortion_measure, initial_codebook=None):
    """
        This method implements Lloyd algorithm. There are two options of initial reconstruct alphabet: (1) begining a unitary codebook and duplicate it in each round. The number of rounds is log2(num_of_levels). And (2) randomized initial reconstruct alphabet from samples.
    """
    if initial_alphabet_opt == 'unitary_until_num_of_elements':
        cw0 = initial_codebook # The inicial unitary codebook is a average of all samples
        cw0_shape = np.shape(cw0)
        codebook = []    
        codebook.append(cw0)
        codebook = np.array(codebook)
        perturbation_variance = 1.0
        perturbation_vector = np.sqrt(perturbation_variance/2) * (np.random.randn(cw0_shape[0], cw0_shape[1]) + 1j * np.random.randn(cw0_shape[0], cw0_shape[1]))
        num_of_rounds = int(np.log2(num_of_levels))

    elif initial_alphabet_opt == 'random_from_samples':
        codebook = initial_codebook
        num_of_rounds = 1 # for randomized initial alphabet method only one round is needed

    elif initial_alphabet_opt == 'random':
        codebook = initial_codebook
        num_of_rounds = 1 # for randomized initial alphabet method only one round is needed

    elif initial_alphabet_opt == 'sa':
        codebook = initial_codebook
        num_of_rounds = 1 # for randomized initial alphabet method only one round is needed
       
    elif initial_alphabet_opt == 'katsavounidis':
        codebook = initial_codebook
        num_of_rounds = 1 # for initial alphabet from user method only one round is needed
 
    elif initial_alphabet_opt == 'xiaoxiao':
        codebook = initial_codebook
        num_of_rounds = 1 # for initial alphabet from user method only one round is needed

    else:
        raise ValueError(f'initial alphabet opt must be one of available opts in \'profile.json\' file')

    mean_distortion_by_round = {}
    current_codebook_dict = None
    mean_distortion_by_iteractions = None


    for r in range(1, num_of_rounds+1):
        if initial_alphabet_opt == 'unitary_until_num_of_elements':
            codebook = duplicate_codebook(codebook, perturbation_vector)

        samples_dict = matrix2dict(samples)
        mean_distortion_by_iteractions = [] #np.zeros(num_of_iteractions)

        for n in range(num_of_iteractions):

            codebook_dict = matrix2dict(codebook)

            sets = {}  # Storage information of partitions baised by each codewords
            for cw_id in codebook_dict.keys():
                sets[cw_id] = []

            distortion = 0  # Distortion measurement of this interaction
            for sample_id, sample in samples_dict.items():
                cw_id, estimated_distortion = perform_distortion(sample, codebook_dict, distortion_measure)
                distortion = distortion + estimated_distortion
                sample_info = {'sample_id': sample_id, 'est_distortion': estimated_distortion}
                sets[cw_id].append(sample_info)
            mean_distortion = distortion/len(samples) 
            mean_distortion_by_iteractions.append(mean_distortion)
            #print (f'iter: {n}, mean_distortion: {mean_distortion}')

            current_codebook_dict = codebook_dict.copy()            
            if (n>0) and (mean_distortion_by_iteractions[n-1] == mean_distortion_by_iteractions[n]):
                break
 

            # Designing a new codebook from sets
            new_codebook_dict = {}
            for cw_id, samples_info_list in sets.items():
                if len(samples_info_list) > 0:
                    samples_sorted = sorted(samples_info_list, key=lambda k: k['est_distortion'])
                    #print ([sample_info['est_distortion'] for sample_info in samples_sorted])
                    sub_set_of_samples = {}
                    for sample_info in samples_sorted:
                        sample_id = sample_info['sample_id']
                        sub_set_of_samples[sample_id] = samples_dict[sample_id]
                    if len(sub_set_of_samples) > 2:
                        sub_set_of_samples_matrix = dict2matrix(sub_set_of_samples) 
                        if distortion_measure == 'mse':
                            new_cw = complex_average(sub_set_of_samples_matrix)
                            new_cw = num_of_levels * new_cw/norm(new_cw) #complex_average(sub_set_of_samples_matrix[start:end])
                        else:
                            raise ValueError(f'Error: no distortion measure option chosen')
                    else:
                        new_cw = complex_average(dict2matrix(sub_set_of_samples))
                        new_cw = num_of_levels * new_cw/norm(new_cw)
                else:
                    if initial_alphabet_opt == 'random_from_samples' or initial_alphabet_opt == 'random' or initial_alphabet_opt == 'sa' or initial_alphabet_opt == 'katsavounidis' or initial_alphabet_opt == 'xiaoxiao':
                        new_cw_index = np.random.choice(len(samples))
                        new_cw = np.array(samples[new_cw_index]) # 
                        #new_cw = new_cw/norm(new_cw)

                    elif initial_alphabet_opt == 'unitary_until_num_of_elements':
                        new_cw = np.array(cw0)
                        #new_cw = new_cw/norm(new_cw)

                new_codebook_dict[cw_id] = new_cw
            codebook = dict2matrix(new_codebook_dict)
        #plot_codebook(codebook, 'designed_codebook_from_round'+str(r)+'.png')
        mean_distortion_by_round[r] = mean_distortion_by_iteractions

    #return dict2matrix(current_codebook_dict), sets,  mean_distortion_by_round
    return current_codebook_dict, sets,  mean_distortion_by_round

# Some plot functions

def plot_unitary_codebook(codebook, filename):
    nrows, ncols = codebook.shape
    fig, axes = plt.subplots(nrows, ncols, subplot_kw=dict(polar=True))
    for col in range(ncols):
        for row in range(nrows):
            a = np.angle(codebook[row,col])
            r = np.abs(codebook[row,col])
            if nrows == 1:
                axes[col].plot(0, 1, 'wo')
                axes[col].plot(a, r, 'ro')
            else:
                axes[row, col].plot(0, 1, 'wo')
                axes[row, col].plot(a, r, 'ro')
    plt.savefig(filename)

def plot_codebook(codebook, filename):
    ncodewords, nrows, ncols = codebook.shape
    #nrows, ncols = codebook.shape
    fig, axes = plt.subplots(ncodewords, ncols, subplot_kw=dict(polar=True))
    #fig, axes = plt.subplots(1, ncols, subplot_kw=dict(polar=True))
    #print (axes.shape)
    for col in range(ncols):
        for cw in range(ncodewords):
            a = np.angle(codebook[cw, 0, col])
            r = np.abs(codebook[cw, 0, col])
            axes[cw, col].plot(0, 1, 'wo')
            axes[cw, col].plot(a, r, 'ro')
    plt.savefig(filename)

def plot_polar_samples(samples, filename):
    nsamples, nrows, ncols = samples.shape
    fig, axes = plt.subplots(nrows, ncols, subplot_kw=dict(polar=True))

    for n in range(nsamples):
        for col in range(ncols):
            a = np.angle(samples[n, 0, col])
            r = np.abs(samples[n, 0, col])
            axes[col].plot(a, r, 'o')
    plt.savefig(filename)

def plot_samples(samples, filename, title, y_label):
    fig, ax = plt.subplots()
    #print (samples)
    #nsamples, nrows, ncols = samples.shape
    #x = np.arange(nsamples)
    #y = samples
    ax.scatter(x=np.arange(len(samples)), y=np.abs(samples), marker='o', c='r', edgecolor='b')
    ax.set_xlabel('samples')
    ax.set_ylabel(y_label)
    plt.title(title)
    plt.savefig(filename)



# JSON STUFF TO ENCODE/DECODE DATA
def encode_codebook(codebook):
    codebook_enc = {}
    for cw_id, cw in codebook.items():
        adjust = {}
        count = 0
        codeword = np.array(cw).reshape(cw.size)
        for complex_adjust in codeword:
            adjust_id = str('complex_adjust') + str(count)
            adjust[adjust_id] = (complex_adjust.real, complex_adjust.imag)
            count += 1
        codebook_enc[str(cw_id)] = adjust
    return codebook_enc

def encode_sets(sets):
    sets_enc = {}
    for cw_id, samples_id_list in sets.items():
        sets_enc[str(cw_id)] = len(samples_id_list)
    return sets_enc

def encode_mean_distortion(distortion_by_round):
    distortion_enc = {}
    for r, distortion_by_interactions in distortion_by_round.items():
        count = 0
        distortion_by_interactions_enc = {}
        for d in distortion_by_interactions:
            distortion_by_interactions_enc[str(count)] = float(d)
            count += 1
        distortion_enc[str(r)] = distortion_by_interactions_enc
    return distortion_enc

def decode_codebook(codebook_json):
    codebook_dict = {}
    for cw_id, cw in codebook_json.items():
        codeword = []
        for cw_adjust in cw.items():
            real_adjust = cw_adjust[1][0]
            imag_adjust = cw_adjust[1][1]
            adjust = real_adjust + 1j * imag_adjust
            codeword.append(adjust)
        codebook_dict[cw_id] = np.array(codeword, dtype=complex)
    return codebook_dict

#def save_training_samples(samples):
#    np.save('samples.npy', samples)

#def load_samples(filename):
#    samples = np.load(filename)
#    return samples


#def std_deviation(vector):
#    de_meaned = vector - average(vector)
#    return norm(de_meaned) * 1/np.sqrt(len(vector))

#def rms():
#    return np.sqrt(np.power(average(x), 2) + np.power(std_deviation(x), 2))

#def complex_correlation(cw1, cw2):
#    cw1 = cw1/norm(cw1)
#    cw2 = cw2/norm(cw2)
#    u = np.matrix([np.real(cw1), np.imag(cw1)])
#    u_vec = np.array(u).reshape(np.size(u))
#    v = np.matrix([np.real(cw2), np.imag(cw2)])
#    v_vec = np.array(v).reshape(np.size(v))
#    correlation = np.inner(u_vec, v_vec)
#    return correlation

#def correlation_factor(x, y):
#    de_meaned_x = x - average(x)
#    de_meaned_y = y - average(y)
#    return np.inner(de_meaned_x, de_meaned_y) / (norm(de_meaned_x) * norm(de_meaned_y))

#def get_mean_distortion(sets, samples, codebook):
#    sum_squared_error = 0
#    for cw_id, samples_id_list in sets.items():
#        cw = codebook[cw_id]
#        for sample_id in samples_id_list:
#            sample = samples[sample_id]    
#            squared_error = np.sum(complex_squared_error(cw, sample))
#            sum_squared_error += squared_error
#    return sum_squared_error/len(samples)

def plot_performance(distortion_by_round, graph_title, filename):
    fig, ax = plt.subplots()
    for r, mean_distortion in distortion_by_round.items():
        ax.plot(mean_distortion, label='#cw: ' + str(2**r))
    plt.ylabel('distortion (MSE)')
    plt.xlabel('# iterations')
    plt.title(graph_title)
    plt.legend()
    fig.savefig(filename)

def hadamard_transform(samples, inverse=False):
    num_samples, num_rows, num_cols = np.shape(samples)
    hadamard_mat = hadamard(int(num_rows * num_cols), dtype=complex)
    samples_converted = []
    channel_size = num_rows * num_cols
    for s in samples:
        s = s.reshape(channel_size)
        s_h = np.zeros((channel_size), dtype=complex)
        for n in range(channel_size):
            s_h[n] = np.sum(hadamard_mat[n].conj() * s)
        if inverse:
            s_h = np.array(s_h).reshape(1, channel_size) * (1/channel_size)
        else:
            s_h = np.array(s_h).reshape(1, channel_size) 
        
        samples_converted.append(s_h.reshape(num_rows, num_cols))
    samples_converted = np.array(samples_converted)

    return samples_converted
  

def check_files(prefix, episodefiles):
    pathfiles = {}
    for ep_file in episodefiles:
        pathfile = prefix + str('/') + str(ep_file)
        ep_file_status = False
        try:
            current_file = open(pathfile)
            ep_file_status = True
            #print("Sucess.")
        except IOError:
            print("File not accessible: ", pathfile)
        finally:
            current_file.close()

        if ep_file_status:
            ep_file_id = uuid.uuid4()
            pathfiles[ep_file_id] = pathfile
 
    return pathfiles


def decode_mean_distortion(mean_distortion_dict):
    mean_distortion_list = []
    for iteration, mean_distortion in mean_distortion_dict.items():
        mean_distortion_list.append(mean_distortion)
    return mean_distortion_list

def get_confidence_interval(results_values, t):
    """
    [1] Confidence Intervals for Unknown Mean and Unknown Standard 
    Deviation <http://www.stat.yale.edu/Courses/1997-98/101/confint.htm>

    [2] Jain, R.; "The Art of Computer Systems Performance Analysis -
    Techniques for Experimental Design, Measurement, Simulation, and
    Modeling"; 1st edition; John Wiley & Sons, Inc.; 1991.

    """
    mean = np.mean(results_values)
    var = np.mean((results_values - mean) ** 2)
    se = np.sqrt(var) # Standard Error
    upbound = mean + t * se/np.sqrt(len(results_values))
    lowbound = mean - t * se/np.sqrt(len(results_values))
    return [lowbound, mean, upbound]

def get_percentiles(results_values):
    first_percentile = np.percentile(results_values, 25) 
    median = np.percentile(results_values, 50) 
    third_percentile = np.percentile(results_values, 75) 
    iqr = third_percentile - first_percentile
    return first_percentile, median, third_percentile, iqr
