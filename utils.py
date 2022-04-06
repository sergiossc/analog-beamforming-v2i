#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sergiossc@gmail.com
"""
import numpy as np
from numpy.linalg import svd, norm
import uuid
import matplotlib.pyplot as plt
import json
from scipy.linalg import hadamard
from scipy.constants import c
import os
#from mayavi import mlab
import scipy.stats as st


#def get_quantized_cb(not_quantized_cb, phase_shift_resolution):
#    nq = 2 ** phase_shift_resolution  # 2 ** phase shift resolution (in bits)
#    quantized_values =  []
#    for i in range(nq):
#        quantized_values.append(np.exp(1j * 2 * np.pi * (1/nq) * i))
#    
#    quantized_cb = [] # np.matrix(np.zeros((array_size, num_of_levels), dtype=complex))
#    nt, num_of_cw = np.shape(not_quantized_cb)
#
#    for l in range(num_of_cw):
#        pass 
#        #cb = []
#        #for v in range(array_size):
#        cb = np.random.choice(quantized_values, array_size)
#
#        cb = np.matrix(cb).reshape(1,array_size)
#        quantized_random_cb.append(cb)
#
#    quantized_random_cb = np.array(quantized_random_cb)
#    return quantized_random_cb

def compare_filter(user_filter, result_filter):
    result = True
    for k, v in user_filter.items():
        if user_filter[k] == result_filter[k] or user_filter[k] is None:
            pass
        else:
            result = False
    return result

def get_datasetname_from_json_resultfiles(d):
    channel_samples_filename = d['channel_samples_filename']
    index = channel_samples_filename.find("s00")
    ds_name = channel_samples_filename[index:index+4]
    if ds_name == '':
        ds_name = None
    return ds_name

def get_all_result_json_pathfiles(rootdir="results"):
    result_pathfiles_dict = {}
    count = 0
    for root, dirs, files in os.walk(rootdir):
        for name in files:
            if name.endswith((".json")):
                count = count + 1
                pathfile = os.path.join(root, name)
                result_pathfiles_dict[count] = pathfile
    return result_pathfiles_dict

def get_all_possible_filters():
    ds_name_available = ['s000', 's002', 's004', 's006', 's007', 's008', 's009'] 
    rx_tx_array_size_available = [(4,4), (8,8), (16,16), (32,32), (64,64)]
    initial_alphabet_opts_available =  ['katsavounidis', 'xiaoxiao', 'random_from_samples', 'random']
    num_of_levels_opts_available =  [4, 8, 16, 32, 64, 128, 256, 512]

    result_filter_set = {}
    result_filter_counter = {}
    for ds_name_opt in ds_name_available:
        for rx_tx_array_size_opt in rx_tx_array_size_available:
            for initial_alphabet_opt in initial_alphabet_opts_available:
                for num_of_levels_opt in num_of_levels_opts_available:
                    k = str(uuid.uuid4())
                    result_filter = {'ds_name': ds_name_opt, 'rx_array_size': rx_tx_array_size_opt[0], 'tx_array_size': rx_tx_array_size_opt[1], 'initial_alphabet_opt': initial_alphabet_opt, 'num_of_levels':num_of_levels_opt}
                    result_filter_set[k] = result_filter
                    result_filter_counter[k] = 0 # this is just a counter os matches
    pass
    return result_filter_set, result_filter_counter

def plot_beamforming_from_codeword(cw, label_text=None, color_list=None, label_dict=None, name=None, ax=None):
    pass
    #cw = cw * np.exp(1j * np.pi/2)
    
    nrows, ncols = np.shape(cw)
    print (nrows, ncols)
    theta = np.arange(0, 2*np.pi, 0.01)
    omega = np.cos(theta)
    psi_mat = np.zeros((ncols, len(omega)))
    d = 0.5 # half wavelengh
    #for n in range(nrows):
    for col in range(ncols):
        for o in range(len(omega)):
            e_x = get_spatial_signature(nrows, omega[o], d)
            v  = np.abs(cw[:,col].conj().T * e_x)[0,0]
            psi_mat[col,o] = v

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    #axs = axs.flatten()
    #print (np.shape(ax))
    
    for col in range(ncols):
        if (label_text is not None) and (color_list is not None):
            ax.plot(theta, psi_mat[col,:], label=label_dict[f'{label_text[col]}'], color=f'{color_list[col]}')
        else:
            ax.plot(theta, psi_mat[col,:], label=f'{col}')
    plt.legend()
    #plt.show()
    fig_filename = f'pattern-s002-n4-{name}.png'
    plt.savefig(fig_filename, bbox_inches='tight') 
    return ax

def get_frab(complex_mat, b):
    """
    Description: 

    Implementation of Finite Resolution Analog Beamforming

    @article{ding2017noma,
    title={NOMA meets finite resolution analog beamforming in massive MIMO and millimeter-wave networks},
    author={Ding, Zhiguo and Dai, Linglong and Schober, Robert and Poor, H Vincent},
    journal={IEEE Communications Letters},
    volume={21},
    number={8},
    pages={1879--1882},
    year={2017},
    publisher={IEEE}
    }

    Input: 
        channel_vec: a list of complex numbers
        Nq: an int number of supported phase shifters
    Output: a list of complex numbers chosen from phase shifters values
    """
    # Nq is the number of supported phase shifters
    #f = []
    nq = 2 ** b
    discrete_phase_values = np.exp(1j * 2 * np.pi * (1/(nq)) * np.arange(nq))
    #print (discrete_phase_values)
    #print (np.rad2deg(np.angle(discrete_phase_values)))
    #print (np.abs(discrete_phase_values))
    #f =  []
    #for i in range(Nq):
    #    f.append(np.exp(1j * 2 * np.pi * (1/Nq) * i))
    #f_vec = []
    #print (f'>>>> shape channel_vec : {np.shape(channel_vec)}')
    num_row, num_col = np.shape(complex_mat)
    complex_mat_discrete = np.matrix(np.zeros((num_row, num_col), dtype=complex))

    for l in range(num_row):
        for k in range(num_col):
            f_min = None
            v_min = 1000 #np.inf
            mat_v = complex_mat[l,k]
            #print (f'(l, k) = {l, k}\tmat_v: {mat_v} \tnorm: {norm(mat_v)}')
            mat_v_norm = norm(mat_v)
            mat_v = mat_v/mat_v_norm

            for f_value in discrete_phase_values:
                pass
                v1 = f_value - mat_v
                #v_squared_norm = np.abs(v1 * v1.conjugate())  # It is just squared error
                v_squared_norm = np.abs(v1) ** 2  # It is just squared error
                if v_squared_norm < v_min:
                    f_min = f_value
                    v_min = v_squared_norm
            complex_mat_discrete[l,k] = f_min * mat_v_norm
            #print (f'(l, k) = {l, k}\tf_min * mat_v_norm: {f_min * mat_v_norm} \tnorm: {norm(f_min * mat_v_norm)}\n')
    return (complex_mat_discrete)
    #return (discrete_phase_values)




def get_quantized_random_cb(array_size, num_of_levels, phase_shift_resolution):
    nq = 2 ** phase_shift_resolution  # 2 ** phase shift resolution (in bits)
    quantized_values =  []
    for i in range(nq):
        quantized_values.append(np.exp(1j * 2 * np.pi * (1/nq) * i))
    
    quantized_random_cb = [] # np.matrix(np.zeros((array_size, num_of_levels), dtype=complex))

    for l in range(num_of_levels):
        pass 
        #cb = []
        #for v in range(array_size):
        cb = np.random.choice(quantized_values, array_size)

        cb = np.matrix(cb).reshape(1,array_size)
        quantized_random_cb.append(cb)

    quantized_random_cb = np.array(quantized_random_cb)
    return quantized_random_cb



def get_beamforming_gain(ch):
    u, s, vh = svd(ch)
    s = s ** 2
    return s[0]
    #return np.sum(s)


def angular_to_array_domain(ch_angular):
    nr, nt = np.shape(ch_angular)
    print (f'nr, nt: {nr, nt}')
    fft_rx, ifft_rx = fftmatrix(nr)
    fft_tx, ifft_tx = fftmatrix(nt)
    ch_array = np.matmul(ifft_rx, ch_angular)
    ch_array = np.matmul(ch_array, fft_tx)
    return ch_array


def array_to_angular_domain(ch_array):
    nr, nt = np.shape(ch_array)
    print (f'nr, nt: {nr, nt}')
    fft_rx, ifft_rx = fftmatrix(nr)
    fft_tx, ifft_tx = fftmatrix(nt)
    ch_angular = np.matmul(fft_rx, ch_array)
    ch_angular = np.matmul(ch_angular, ifft_tx)
    return ch_angular



def fftmatrix(n, q=None):
    """
    The FFT matrix is from:
    @book{brunton2019data,
     title={Data-driven science and engineering: Machine learning, dynamical systems, and control},
     author={Brunton, Steven L and Kutz, J Nathan},
     year={2019},
     publisher={Cambridge University Press}
    }

    The oversampling factor q if from:
    
    @book{zaidi20185g,
    title={5G Physical Layer: principles, models and technology components},
    author={Zaidi, Ali and Athley, Fredrik and Medbo, Jonas and Gustavsson, Ulf and Durisi, Giuseppe and Chen, Xiaoming},
    year={2018},
    publisher={Academic Press}
    }
    """
    if  q is None:
        w = np.exp(-1j * 2 * np.pi / n)
        fftmat = np.matrix(np.zeros((n,n), dtype=complex))
        for l in range(n):
            for k in range(n):
                fftmat[k,l] = w ** (l*k)
        fftmat = fftmat/np.sqrt(n)
        ifftmat = np.conj(fftmat)
    else:
        w = np.exp(-1j * 2 * np.pi / (q*n))
        fftmat = np.matrix(np.zeros((n,q*n), dtype=complex))
        for l in range(n):
            for k in range(q*n):
                fftmat[l,k] = w ** (l*k)
        fftmat = fftmat.T
        fftmat = fftmat/np.sqrt(n)
        ifftmat = np.conj(fftmat)
    
    return fftmat, ifftmat


def get_codebook(pathfile):
    with open(pathfile) as result:
        data = result.read()
        d = json.loads(data)
    return decode_codebook(d['codebook'])

def beamsweeping(ch, cb_dict):

    p_est_max = -np.Inf
    cw_id_max_tx = ''
    cw_id_max_rx = ''
    n_rx, n_tx = np.shape(ch)
    
    for k in cb_dict.keys():
        cw_tx = cb_dict[k]
        cw_tx = np.matrix(cw_tx)
        #print (f'shape of cw_tx: {np.shape(cw_tx)}')
        #cw_tx = np.array(cw_tx).reshape(n_rx, n_tx)
        #cw_tx = m * cw_tx/norm(cw_tx) # squared norm
        #print (f'norm(cw): {norm(cw)}')
        #u_tx, s_tx, vh_tx = svd(cw_tx)
        #u_s = np.matrix(u_s)
        #vh_tx = np.matrix(vh_tx)
        #f_s = vh_tx[0,:]
        #w_s = u_s[:,0]
        #print (f'-------------> norm of f: {norm(np.array(cw_tx))}')
       
        for l in cb_dict.keys():
            cw_rx = cb_dict[l]
            cw_rx = np.matrix(cw_rx)
            #print (f'shape of cw_rx: {np.shape(cw_rx)}')
            #cw_rx = np.array(cw_rx).reshape(n_rx, n_tx)
            #cw_rx = m * cw_rx/norm(cw_rx) # squared norm
            #print (f'norm(cw): {norm(cw)}')
            #u_rx, s_rx, vh_rx = svd(cw_rx)
            #u_rx = np.matrix(u_rx)
            #vh_s = np.matrix(vh_s)
            #f_s = vh_s[0,:]
            #w_s = u_rx[:,0]
            #print (f'-------------> norm of ws: {norm(np.array(cw_rx))}')

            p_s = cw_rx.conj() * (ch * cw_tx.conj().T)
            #print (p_s)
            p_s = norm(p_s) ** 2
            #p_s = np.abs(p_s.conj() * p_s)
            if p_s > p_est_max:
                p_est_max = p_s
                cw_id_max_tx = k
                cw_id_max_rx = l

    return p_est_max, cw_id_max_tx, cw_id_max_rx


def beamsweeping2(ch, cb_dict):
    '''
        This is correct! Checked!!
    '''

    p_est_max = -np.Inf
    cw_id_max_tx = ''
    n_rx, n_tx = np.shape(ch)
    
    u, s, vh = svd(ch)
    #u = np.matrix(u)
    #vh = np.matrix(vh)
    #f = vh[0,:]
    f = np.matrix(vh[0,:]).T
    #w = u[:,0]
    w = np.matrix(u[:,0]).T
    cw_rx = w
    
    for k in cb_dict.keys():
        cw_tx = cb_dict[k]
        ##cw_tx = np.matrix(cw_tx)
        #print (f'shape of cw_tx: {np.shape(cw_tx)}')
        #cw_tx = np.array(cw_tx).reshape(n_rx, n_tx)
        #cw_tx = m * cw_tx/norm(cw_tx) # squared norm
        #print (f'norm(cw): {norm(cw)}')
        #print (f'-------------> norm of f: {norm(np.array(cw_tx))}')
       
        #for l in cb_dict.keys():
        #    cw_rx = cb_dict[l]
        #    cw_rx = np.matrix(cw_rx)
        #    print (f'shape of cw_rx: {np.shape(cw_rx)}')
        #    #cw_rx = np.array(cw_rx).reshape(n_rx, n_tx)
        #    #cw_rx = m * cw_rx/norm(cw_rx) # squared norm
        #    #print (f'norm(cw): {norm(cw)}')
        #    #u_rx, s_rx, vh_rx = svd(cw_rx)
        #    #u_rx = np.matrix(u_rx)
        #    #vh_s = np.matrix(vh_s)
        #    #f_s = vh_s[0,:]
        #    #w_s = u_rx[:,0]
        #    #print (f'-------------> norm of ws: {norm(np.array(cw_rx))}')
    
        p_s = cw_rx.conj().T * (ch * cw_tx.conj())
        #print (p_s)
        p_s = norm(p_s) ** 2
        #p_s = np.abs(p_s.conj() * p_s)
        if p_s > p_est_max:
            p_est_max = p_s
            cw_id_max_tx = k
            #cw_id_max_rx = l

    return p_est_max, cw_id_max_tx #, cw_id_max_rx



def wf(s, snr_v):
    """
        wterfilling power allocation
        input: 'snr' value(float), 's' eigenvalues (numpy.array)
        
        https://www.mathworks.com/matlabcentral/fileexchange/3592-water-filling-algorithm
    """
    vec = 1/(snr_v * s)
    p_alloc = np.zeros(len(vec))
    tol = 0.0001
    pcon = 1
    n = len(vec)
    wline = np.min(vec) + pcon/n
    ptot = np.sum([np.max([wline - vec[i],0]) for i in range(len(vec)) ])
    while (np.abs(pcon-ptot) > tol):
        wline += (pcon-ptot)/n
        p_alloc = np.array([np.max([wline - vec[i],0]) for i in range(len(vec)) ])
        ptot = np.sum(p_alloc)
    #print (p_alloc)
    return p_alloc 

def siso_c(snr_v):
    c = np.log2(1 + snr_v)
    return c

def mimo_csir_c(s, snr_v):
    c = 0
    for i in range(len(s)):
        c += np.log2(1 + snr_v * (1/len(s)) * s[i])
    return c


def mimo_csit_eigenbeamforming_c(s, snr_v):
    c = 0
    opt_power_alloc = wf(s, snr_v)
    for i in range(len(s)):
        c += np.log2(1 + snr_v * opt_power_alloc[i] * s[i])
    return c

def mimo_csit_beamforming(s, snr_v):
    c = 0
    #print (f's0: {s[0]}')
    c = np.log2(1 + snr_v * s[0])
    return c


#def get_spatial_signature2(n, d, wavelength, omega):
#    spatial_signature = (1/np.sqrt(n)) * np.matrix([np.exp(1j * 2 * np.pi * d/wavelength * i * omega) for i in range(n)])
#    return spatial_signature

def get_spatial_signature(n, omega, d):
    spatial_signature = (1/np.sqrt(n)) * np.matrix([np.exp(-1j * 2 * np.pi * d * i * omega) for i in range(n)])
    return spatial_signature.T

def get_orthonormal_basis(n, d, wavelength):
    L = n * d
    s_set = np.zeros((n,n), dtype=complex)
    #slots = np.arange(-n/2, n/2 + 1)
    slots = np.arange(n)
    for i in range(n):
        e = get_spatial_signature(n, d, wavelength, slots[i]/L)
        s_set[i:] = e
    s_set = np.matrix(s_set)
    return s_set

def plot_angular_domain(nr, Lr, nt, Lt, paths):
    s_rx = np.zeros(nr)
    s_tx = np.zeros(nt)
    for p in paths:
        pathgain = p['pathgain']
        aoa = p['aoa']
        aod = p['aod']
        k_index = None
        l_index = None
        for k in range(nr):
            if np.cos(aoa) > (k/Lr - 1/(2*Lr)) and np.cos(aoa) <= (k/Lr + 1/(2*Lr)):
                k_index = k
        for l in range(nt):
            if np.cos(aod) > (l/Lt - 1/(2*Lt)) and np.cos(aod) <= (l/Lt + 1/(2*Lt)):
                l_index = l
    print (f'k_index = {k_index}')
    print (f'l_index = {l_index}')

def gen_channel(num_rx, num_tx, variance):
    sigma = variance
    h = np.sqrt(sigma/2)*(np.random.randn(num_rx, num_tx) + np.random.randn(num_rx, num_tx) * 1j)
    return h

def gen_hermitian_matrix(dim):
    """
    REF: https://stackoverflow.com/questions/57439865/how-to-generate-a-random-sparse-hermitian-matrix-in-python
    """
    var = 1
    a = np.sqrt(var/2) *  (np.random.rand(dim, dim) + np.random.rand(dim, dim) * 1j)
    a = np.matrix(a)
    a = a + a.conj().T
    a = a/np.linalg.norm(a)
    return a

def squared_norm(cw):
    """
    Input: cw as a vector (1-dim)
    Output: return a squared norm as a inner product of cw.conj() * cw
    """
    #f_cw = cw.flatten()
    #inner_product = np.sum(f_cw.conj() * f_cw) 
    inner_product = norm(cw) ** 2 #np.sum(f_cw.conj() * f_cw) 
    
    return inner_product

#def norm(cw): 
#    return np.sqrt(squared_norm(cw))

def get_precoder_combiner(h):
    u, s, vh = svd(h)
    precoder = np.matrix(vh).conj().T[:,0]
    combining = np.matrix(u).conj().T[0,:]
    return precoder, combining

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

def loschnmtx(complex_gain, rx_array, tx_array, aoa, aod):
    aoa = np.deg2rad(aoa) # convert degto rad
    aod = np.deg2rad(aod) # convert deg to rad
    
    rx_size = rx_array.size
    tx_size = tx_array.size

    #h = np.zeros((rx_size, tx_size), dtype=complex)
    rx_spacing = rx_array.element_spacing
    tx_spacing = tx_array.element_spacing
    
    rx_wavelength = rx_array.wave_length
    tx_wavelength = tx_array.wave_length

    k_rx = 2 * np.pi * (1/rx_wavelength)
    k_tx = 2 * np.pi * (1/tx_wavelength)

    factor = np.sqrt(rx_size * tx_size)

    rx_array_vec = np.arange(rx_size)
    tx_array_vec = np.arange(tx_size)

    ar = np.matrix([np.exp(-1j * k_rx * n * rx_spacing * np.cos(aoa)) for n in range(len(rx_array_vec))])
    ar = (1/np.sqrt(rx_size)) * ar.T

    at = np.matrix([np.exp(-1j * k_tx * n * tx_spacing * np.cos(aod)) for n in range(len(tx_array_vec))])
    at = (1/np.sqrt(tx_size)) * at.T

    product = complex_gain * np.array(ar * at.conj().T)

    h = factor * product

    return np.matrix(h)

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

def sorted_samples_2(samples_dict, attr='norm'):
    nsamples = len(samples_dict) #, nrows, ncols = samples.shape
    s_not_sorted = []

    if attr == 'variance_characteristic_value': # From the paper

        #s_avg = complex_average(samples)
        for k, v in samples_dict.items():
            s = v
            num_rx, num_tx = s.shape
            s_avg = np.sum(s)/num_tx
            s_demeaned = s - s_avg
            #s_var = np.sqrt(np.sum(s_demeaned.conj() * s_demeaned)/num_tx)
            s_var = np.sum(s_demeaned * s_demeaned.conj())/num_tx  # variance of a vector x is given by expected values of (xi - mx)^2 whre xi is i-th element of x
            s_info = {}
            s_info = {'sample_id': k,'s_var': np.abs(s_var), 's': s}
            s_not_sorted.append(s_info)

        s_sorted = sorted(s_not_sorted, key=lambda k: k['s_var']) # 
        #samples_sorted = [v['s'] for v in s_sorted]
        #attr_sorted = [v['s_var'] for v in s_sorted]


    else:
        return -1

    #return np.array(samples_sorted), np.array(attr_sorted)
    return s_sorted #list of dict




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
            #s_var = np.sqrt(np.sum(s_demeaned.conj() * s_demeaned)/num_tx)
            s_var = np.sum(s_demeaned * s_demeaned.conj())/num_tx  # variance of a vector x is given by expected values of (xi - mx)^2 whre xi is i-th element of x
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
        #mse = squared_norm(cw - sample)/np.size(sample)
        #print (f'cw shape: {np.shape(cw)}')
        #print (f'sample shape: {np.shape(sample)}')
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


def xiaoxiao_initial_codebook_2(samples_dict, num_of_levels):
    '''
    @article{ma2015high,
    title={High-quality initial codebook design method of vector quantisation using grouping strategy},
    author={Ma, Xiaoxiao and Pan, Zhibin and Li, Yang and Fang, Jie},
    journal={IET Image Processing},
    volume={9},
    number={11},
    pages={986--992},
    year={2015},
    publisher={IET}
    }
    '''

    #num_samples, num_rows, num_cols = samples.shape
    num_samples = len(samples_dict)

    # Code samples in hadamard code
    #samples_hadamard = hadamard_transform(samples, False)    

    # Ordering samples by variance characteristic value (ascending way)
    #samples_sorted, attr_sorted = sorted_samples(samples_hadamard, 'variance_characteristic_value') 
    samples_sorted = sorted_samples_2(samples_dict, 'variance_characteristic_value') 
    
    # Index A, B and C groups
    a_group_begin = 0
    a_group_end = 1 * int(num_samples/20)

    b_group_begin = a_group_end
    b_group_end = b_group_begin + (17 * int(num_samples/20))

    c_group_begin = b_group_end
    c_group_end = -1 

    # Getting samples from ordered samples spliting in groups as indexed as before
    a_group_of_samples = samples_sorted[a_group_begin:a_group_end]
    b_group_of_samples = samples_sorted[b_group_begin:b_group_end]
    c_group_of_samples = samples_sorted[c_group_begin:c_group_end]
    
    # Ordering subgroups by mean characteristic value
    samples_a_group_sorted = a_group_of_samples #sorted_samples(a_group_of_samples, 'abs_mean_characteristic_value') 
    samples_b_group_sorted = b_group_of_samples #sorted_samples(b_group_of_samples, 'abs_mean_characteristic_value') 
    samples_c_group_sorted = c_group_of_samples #sorted_samples(c_group_of_samples, 'abs_mean_characteristic_value') 

    # For each subgroup, select the codewords. Ex.: all/2, all/4 and all/4 number of codewords
    num_of_codewords = num_of_levels

    print (len(samples_a_group_sorted))
    print (len(samples_b_group_sorted))
    print (len(samples_c_group_sorted))

    #print ('len(group_a): ', len(samples_a_group_sorted))
    index_a = get_index_codewords_from_sub_samples(len(samples_a_group_sorted), num_of_codewords/4)
    print ('index_a:', index_a)

    #print ('len(group_b): ', len(samples_b_group_sorted))
    index_b = get_index_codewords_from_sub_samples(len(samples_b_group_sorted), num_of_codewords/2)
    print ('index_b:', index_b)

    #print ('len(group_c): ', len(samples_c_group_sorted))
    index_c = get_index_codewords_from_sub_samples(len(samples_c_group_sorted), num_of_codewords/4)
    print ('index_c:', index_c)


    #igetting codewords from subgroups
    list_initial_codebook_from_a_group = [samples_a_group_sorted[i] for i in index_a]
    list_initial_codebook_from_b_group = [samples_b_group_sorted[i] for i in index_b]
    list_initial_codebook_from_c_group = [samples_c_group_sorted[i] for i in index_c]

    initial_codebook = np.array(list_initial_codebook_from_a_group + list_initial_codebook_from_b_group + list_initial_codebook_from_c_group)

    #print (initial_codebook.shape)
    #return initial_codebook, samples_hadamard
    return initial_codebook






def xiaoxiao_initial_codebook(samples, num_of_levels):
    '''
    @article{ma2015high,
    title={High-quality initial codebook design method of vector quantisation using grouping strategy},
    author={Ma, Xiaoxiao and Pan, Zhibin and Li, Yang and Fang, Jie},
    journal={IET Image Processing},
    volume={9},
    number={11},
    pages={986--992},
    year={2015},
    publisher={IET}
    }

    '''

    num_samples, num_rows, num_cols = samples.shape

    # Code samples in hadamard code
    #samples_hadamard = hadamard_transform(samples, False)    

    # Ordering samples by variance characteristic value (ascending way)
    #samples_sorted, attr_sorted = sorted_samples(samples_hadamard, 'variance_characteristic_value') 
    samples_sorted, attr_sorted = sorted_samples(samples, 'variance_characteristic_value') 
    
    # Index A, B and C groups
    a_group_begin = 0
    a_group_end = 1 * int(num_samples/20)

    b_group_begin = a_group_end
    b_group_end = b_group_begin + (2 * int(num_samples/20))

    c_group_begin = b_group_end
    c_group_end = -1 

    # Getting samples from ordered samples spliting in groups as indexed as before
    a_group_of_samples = samples_sorted[a_group_begin:a_group_end, :, :]
    b_group_of_samples = samples_sorted[b_group_begin:b_group_end, :, :]
    c_group_of_samples = samples_sorted[c_group_begin:c_group_end, :, :]
    
    # Ordering subgroups by mean characteristic value
    samples_a_group_sorted = a_group_of_samples # sorted_samples(a_group_of_samples, 'abs_mean_characteristic_value') 
    samples_b_group_sorted = b_group_of_samples # sorted_samples(b_group_of_samples, 'abs_mean_characteristic_value') 
    samples_c_group_sorted = c_group_of_samples # sorted_samples(c_group_of_samples, 'abs_mean_characteristic_value') 

    # For each subgroup, select the codewords. Ex.: all/2, all/4 and all/4 number of codewords
    num_of_codewords = num_of_levels

    #print ('len(group_a): ', len(samples_a_group_sorted))
    index_a = get_index_codewords_from_sub_samples(len(samples_a_group_sorted), num_of_codewords/4)
    #print ('index_a:', index_a)

    #print ('len(group_b): ', len(samples_b_group_sorted))
    index_b = get_index_codewords_from_sub_samples(len(samples_b_group_sorted), num_of_codewords/2)
    #print ('index_b:', index_b)

    #print ('len(group_c): ', len(samples_c_group_sorted))
    index_c = get_index_codewords_from_sub_samples(len(samples_c_group_sorted), num_of_codewords/4)
    #print ('index_c:', index_c)


    #igetting codewords from subgroups
    list_initial_codebook_from_a_group = [samples_a_group_sorted[i] for i in index_a]
    list_initial_codebook_from_b_group = [samples_b_group_sorted[i] for i in index_b]
    list_initial_codebook_from_c_group = [samples_c_group_sorted[i] for i in index_c]

    initial_codebook = np.array(list_initial_codebook_from_a_group + list_initial_codebook_from_b_group + list_initial_codebook_from_c_group)

    #print (initial_codebook.shape)
    #return initial_codebook, samples_hadamard
    return initial_codebook

def get_index_codewords_from_sub_samples(n_samples, n_codewords):

    slot = int(n_samples/n_codewords)
    step = slot/2

    index_codebook_list = []

    for n in range(int(n_codewords)):
            start = n * slot
            mid = start + step
            index_codebook_list.append(int(mid))
    return index_codebook_list

def katsavounidis_initial_codebook_2(samples, num_levels):
    '''
    Katsavounidis, I., Kuo, C.C.J., Zhang, Z.: ‘A new initialization technique for
    generalized Lloyd algorithm’, IEEE Signal Process. Lett., 1994, 1, (10),
    pp. 144–146
    
    '''
    num_samples, num_rows, num_cols = samples.shape


    samples_dict = matrix2dict(samples)
    samples_dict_keys_list = list(samples_dict.keys())
    #samples_dict_copy = samples_dict.copy()
        
    #max_norm = -np.Inf
    min_complex_inner_product = 100
    min_sample_id_1 = ''
    min_sample_id_2 = ''

    all_values = {}

    for i in range(len(samples_dict_keys_list)):
        if i < (len(samples_dict_keys_list)):
            ki = samples_dict_keys_list[i]
            for j in range(i+1, len(samples_dict_keys_list)):
    #for k in samples_dict_keys_list:
                kj = samples_dict_keys_list[j]
                #print(k)
                #print ('--->')
                prod = np.abs(np.sum(samples_dict[ki] * samples_dict[kj].conj()))
                all_values[(ki, kj)] = prod
                #print (np.shape(samples_dict[ki]))
                #print (np.shape(samples_dict[kj]))
                #print (prod)
                #if prod < min_complex_inner_product:
                #    #print (prod)
                #    min_complex_inner_product = prod
                #    min_sample_id_1 = ki
                #    min_sample_id_2 = kj
    #print (f'min_complex_inner_product: {min_complex_inner_product}')
    #print (f'min_sample_id_1: {samples_dict[min_sample_id_1]}')
    #print (f'min_sample_id_2: {samples_dict[min_sample_id_2]}')
    for k, v in all_values.items():
        print (f'k: {k}\n')
        print (f'v: {v}\n')
        
    print (len(all_values))
    
    # find the sampla with max norm as first cw 
    #for s_id, s in samples_dict.items():
    #    pass
        #s_norm = norm(s)
        #print (f'----------------> s_norm: {s_norm}')
#        if s_norm > max_norm:
#            max_norm = s_norm
#            max_sample_id = s_id
#    
#    #num_of_codewords = num_cols
#    num_of_codewords = num_levels
#    initial_codebook = np.zeros((num_of_codewords, num_rows, num_cols), dtype=complex)
    
#    # Remove the max_sample_id from samples_dict and add it as our first codeword in initial_codebook
#    initial_codebook[0,:,:] = samples_dict.pop(max_sample_id) 
#
#    # Step 2: Define 2nd codeword as the largest distance from the 1st codeword
#    cw = initial_codebook[0,:,:]
#    max_distance = -np.Inf
#    max_distance_sample_id = '' 
#    for s_id, s in samples_dict.items():
#        s_distance = norm(s - cw)
#        if s_distance > max_distance:
#            max_distance = s_distance
#            max_distance_sample_id = s_id
#    initial_codebook[1,:,:] = samples_dict.pop(max_distance_sample_id)
#
#    # Step 3: defining next codewords
#
#    for i in range(0, num_of_codewords - 2):
#
#        min_distance = np.Inf
#        min_distance_sample_id = '' 
#
#        for s_id, s in samples_dict.items():
#            s_distance = 0
#            for codeword in initial_codebook:
#                s_distance = s_distance + norm(s - codeword)
#            if s_distance < min_distance:
#                min_distance = s_distance
#                min_distance_sample_id = s_id
#    
#        #for s_id, s in samples_dict.items():
#        #    for codeword in initial_codebook:
#        #        s_distance = norm(s - codeword)
#        #        if s_distance < min_distance:
#        #            min_distance = s_distance
#        #            min_distance_sample_id = s_id
#    
#        max_distance = -np.Inf
#        max_distance_sample_id = '' 
#
#        for s_id, s in samples_dict.items():
#    
#            s_distance = norm(s - samples_dict[min_distance_sample_id])
#            if s_distance > max_distance:
#                max_distance = s_distance
#                max_distance_sample_id = s_id
#    
#    
#        initial_codebook[i+2,:,:] = samples_dict.pop(max_distance_sample_id)
#
#    initial_codebook_normalized = np.zeros((num_of_codewords, num_rows, num_cols), dtype=complex)
#    for i in range(num_of_codewords):
#        initial_codebook_normalized[i,:,:] = initial_codebook[i,:,:]/norm(initial_codebook[i,:,:]) 
#
#    return initial_codebook_normalized
 

def katsavounidis_initial_codebook(samples, num_levels):
    '''
    Katsavounidis, I., Kuo, C.C.J., Zhang, Z.: ‘A new initialization technique for
    generalized Lloyd algorithm’, IEEE Signal Process. Lett., 1994, 1, (10),
    pp. 144–146
    
    '''
    num_samples, num_rows, num_cols = samples.shape


    samples_dict = matrix2dict(samples)
        
    max_norm = -np.Inf
    max_sample_id = ''

    # find the sampla with max norm as first cw 
    for s_id, s in samples_dict.items():
        s_norm = norm(s)
        if s_norm > max_norm:
            max_norm = s_norm
            max_sample_id = s_id
    
    #num_of_codewords = num_cols
    num_of_codewords = num_levels
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

#def run_lloyd_gla(parm):
#    print (parm.keys())
#
#    #dict_keys(['channel_samples', 'initial_alphabet_opt', 'distortion_measure_opt', 'max_num_of_interactions', 'results_dir', 'trial_id', 'trial_random_seed'])
#
#    data = {}
#
#    trial_id = parm['trial_id']
#    data['trial_id'] = trial_id
#
#    results_dir = parm['results_dir']
#    data['results_dir'] = results_dir
#
#    #json_filename = str(results_dir) + '/' + str(instance_id) + '.json'
#    json_filename = str(results_dir) + f'/result_{trial_id}.json'
#
#    initial_alphabet_opt = parm['initial_alphabet_opt']
#    data['initial_alphabet_opt'] = initial_alphabet_opt
#
#    distortion_measure_opt = parm['distortion_measure_opt']
#    data['distortion_measure_opt'] = distortion_measure_opt
#
#    max_num_of_interactions = parm['max_num_of_interactions']
#    data['max_num_of_interactions'] = max_num_of_interactions
#
#    trial_random_seed = parm['trial_random_seed']
#    data['trial_random_seed'] = float(trial_random_seed)
#
#    samples = parm['channel_samples']
#    #samples = samples[0:100]
#    num_of_samples, rx_array_size, tx_array_size = np.shape(samples)
#    data['rx_array_size'] = rx_array_size
#    data['tx_array_size'] = tx_array_size
#    data['num_of_samples'] = num_of_samples
#    num_of_levels = tx_array_size + 1
#    data['num_of_levels'] = num_of_levels 
#
#    # Starting lloyd with an specific initial alphabet opt
#
#    if initial_alphabet_opt == 'katsavounidis':
#        pass 
#    
#    elif initial_alphabet_opt == 'xiaoxiao':
#        pass 
#
#    elif initial_alphabet_opt == 'sa':
#        pass 
#
#    elif initial_alphabet_opt == 'unitary_until_num_of_elements':
#        pass 
#
#    elif initial_alphabet_opt == 'random_from_samples':
#        initial_codebook = np.array([samples[i] for i in np.random.choice(len(samples), num_of_levels, replace=False)])
#        codebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)
#
#    elif initial_alphabet_opt == 'random':
#        pass 
#
#    else:
#        print (f'initial_alphabet_opt \'{initial_alphabet_opt}\' not found')
#        exit()
##    if initial_alphabet_opt == 'xiaoxiao':
##        initial_codebook, samples_hadamard = xiaoxiao_initial_codebook(samples)
##        codebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)
##
##    elif initial_alphabet_opt == 'katsavounidis':
##        initial_codebook = katsavounidis_initial_codebook(samples)
##        codebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)
##
##    elif initial_alphabet_opt == 'sa':
##        initial_codebook = np.array([samples[i] for i in np.random.choice(len(samples), num_of_levels, replace=False)])
##        initial_temperature = 10
##        sa_max_num_of_iteractions = 20
##        variance_of_samples = 1.0 ######
##        codebook, sets, mean_distortion_by_round = sa(initial_codebook, variance_of_samples, initial_temperature, sa_max_num_of_iteractions, max_num_of_interactions, distortion_measure_opt, num_of_levels, samples)
##
##    elif initial_alphabet_opt == 'unitary_until_num_of_elements':
##        initial_codebook = complex_average(samples)
##        initial_codebook = initial_codebook/norm(initial_codebook)
##        codebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)
##
##    elif initial_alphabet_opt == 'random_from_samples':
##        pass
##        #initial_codebook = np.array([samples[i] for i in np.random.choice(len(samples), num_of_levels, replace=False)])
##        #codebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)
# 
##    elif initial_alphabet_opt == 'random':
##        angle_range = np.linspace(0, 2*np.pi, 360)
##        initial_codebook = num_of_levels/np.sqrt(rx_array_size * tx_array_size) *  np.exp(1j * np.random.choice(angle_range, (num_of_levels, rx_array_size, tx_array_size), replace=True))
##        codebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)
##        
##
#    #data['initial_codebook'] = encode_codebook(matrix2dict(initial_codebook))
#    #data['codebook'] = encode_codebook(matrix2dict(codebook))
#    data['codebook'] = encode_codebook(codebook)
#    ##plot_performance(mean_distortion_by_round, 'MSE as distortion', 'distortion.png')
#    #codebook = 1/np.sqrt(num_of_elements) * np.exp(1j * np.angle(codebook))
#    #data['egt_codebook'] = encode_codebook(matrix2dict(lloydcodebook))
#    data['sets'] = encode_sets(sets)
#    data['mean_distortion_by_round'] = encode_mean_distortion(mean_distortion_by_round)
#
#    with open(json_filename, "w") as write_file:
#        json.dump(data, write_file, indent=4)
#
#    return 0
#
#def lloyd_gla(initial_alphabet_opt, samples, num_of_levels, num_of_iteractions, distortion_measure, initial_codebook=None):
#    """
#        This method implements Lloyd algorithm. There are two options of initial reconstruct alphabet: (1) begining a unitary codebook and duplicate it in each round. The number of rounds is log2(num_of_levels). And (2) randomized initial reconstruct alphabet from samples.
#    """
#    if initial_alphabet_opt == 'unitary_until_num_of_elements':
#        cw0 = initial_codebook # The inicial unitary codebook is a average of all samples
#        cw0_shape = np.shape(cw0)
#        codebook = []    
#        codebook.append(cw0)
#        codebook = np.array(codebook)
#        perturbation_variance = 1.0
#        perturbation_vector = np.sqrt(perturbation_variance/2) * (np.random.randn(cw0_shape[0], cw0_shape[1]) + 1j * np.random.randn(cw0_shape[0], cw0_shape[1]))
#        num_of_rounds = int(np.log2(num_of_levels))
#
#    elif initial_alphabet_opt == 'random_from_samples':
#        codebook = initial_codebook
#        num_of_rounds = 1 # for randomized initial alphabet method only one round is needed
#
#    elif initial_alphabet_opt == 'random':
#        codebook = initial_codebook
#        num_of_rounds = 1 # for randomized initial alphabet method only one round is needed
#
#    elif initial_alphabet_opt == 'sa':
#        codebook = initial_codebook
#        num_of_rounds = 1 # for randomized initial alphabet method only one round is needed
#       
#    elif initial_alphabet_opt == 'katsavounidis':
#        codebook = initial_codebook
#        num_of_rounds = 1 # for initial alphabet from user method only one round is needed
# 
#    elif initial_alphabet_opt == 'xiaoxiao':
#        codebook = initial_codebook
#        num_of_rounds = 1 # for initial alphabet from user method only one round is needed
#
#    else:
#        raise ValueError(f'initial alphabet opt must be one of available opts in \'profile.json\' file')
#
#    mean_distortion_by_round = {}
#    current_codebook_dict = None
#    mean_distortion_by_iteractions = None
#
#
#    for r in range(1, num_of_rounds+1):
#        if initial_alphabet_opt == 'unitary_until_num_of_elements':
#            codebook = duplicate_codebook(codebook, perturbation_vector)
#
#        samples_dict = matrix2dict(samples)
#        mean_distortion_by_iteractions = [] #np.zeros(num_of_iteractions)
#
#        for n in range(num_of_iteractions):
#
#            codebook_dict = matrix2dict(codebook)
#
#            sets = {}  # Storage information of partitions baised by each codewords
#            for cw_id in codebook_dict.keys():
#                sets[cw_id] = []
#
#            distortion = 0  # Distortion measurement of this interaction
#            for sample_id, sample in samples_dict.items():
#                cw_id, estimated_distortion = perform_distortion(sample, codebook_dict, distortion_measure)
#                distortion = distortion + estimated_distortion
#                sample_info = {'sample_id': sample_id, 'est_distortion': estimated_distortion}
#                sets[cw_id].append(sample_info)
#            mean_distortion = distortion/len(samples) 
#            mean_distortion_by_iteractions.append(mean_distortion)
#            #print (f'iter: {n}, mean_distortion: {mean_distortion}')
#
#            current_codebook_dict = codebook_dict.copy()            
#            if (n>0) and (mean_distortion_by_iteractions[n-1] == mean_distortion_by_iteractions[n]):
#                break
# 
#
#            # Designing a new codebook from sets
#            new_codebook_dict = {}
#            for cw_id, samples_info_list in sets.items():
#                if len(samples_info_list) > 0:
#                    samples_sorted = sorted(samples_info_list, key=lambda k: k['est_distortion'])
#                    #print ([sample_info['est_distortion'] for sample_info in samples_sorted])
#                    sub_set_of_samples = {}
#                    for sample_info in samples_sorted:
#                        sample_id = sample_info['sample_id']
#                        sub_set_of_samples[sample_id] = samples_dict[sample_id]
#                    if len(sub_set_of_samples) > 2:
#                        sub_set_of_samples_matrix = dict2matrix(sub_set_of_samples) 
#                        if distortion_measure == 'mse':
#                            new_cw = complex_average(sub_set_of_samples_matrix)
#                            new_cw = num_of_levels * new_cw/norm(new_cw) #complex_average(sub_set_of_samples_matrix[start:end])
#                        else:
#                            raise ValueError(f'Error: no distortion measure option chosen')
#                    else:
#                        new_cw = complex_average(dict2matrix(sub_set_of_samples))
#                        new_cw = num_of_levels * new_cw/norm(new_cw)
#                else:
#                    if initial_alphabet_opt == 'random_from_samples' or initial_alphabet_opt == 'random' or initial_alphabet_opt == 'sa' or initial_alphabet_opt == 'katsavounidis' or initial_alphabet_opt == 'xiaoxiao':
#                        new_cw_index = np.random.choice(len(samples))
#                        new_cw = np.array(samples[new_cw_index]) # 
#                        #new_cw = new_cw/norm(new_cw)
#
#                    elif initial_alphabet_opt == 'unitary_until_num_of_elements':
#                        new_cw = np.array(cw0)
#                        #new_cw = new_cw/norm(new_cw)
#
#                new_codebook_dict[cw_id] = new_cw
#            codebook = dict2matrix(new_codebook_dict)
#        #plot_codebook(codebook, 'designed_codebook_from_round'+str(r)+'.png')
#        mean_distortion_by_round[r] = mean_distortion_by_iteractions
#
#    #return dict2matrix(current_codebook_dict), sets,  mean_distortion_by_round
#    return current_codebook_dict, sets,  mean_distortion_by_round

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

def AntennaPartern(tx_array, w):
    theta = np.arange(0,180)
   
    num_tx_elements = tx_array.size
    wavelength = tx_array.wave_length  #in meters
    element_spacing = tx_array.element_spacing #in meters
    tx_array_vec = np.arange(num_tx_elements)
   
    at = w
    prod = np.zeros(len(theta), dtype=complex)
    for i in range(len(theta)):
        t = np.deg2rad(theta[i])
        af = np.array([np.exp(1j * 2 * np.pi * n * element_spacing * (1/wavelength) * np.cos(t)) for n in range(len(tx_array_vec))])
        product = np.matmul(at.T, af)
        product = (np.abs(product) ** 2)/np.abs(np.matmul(at.conj().T, at))#print ('product: \n', product)
        prod[i] = product/num_tx_elements

    #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    plt.plot(theta, np.abs(prod), label='w')
    plt.legend()
    #plt.savefigfig('steering.png')
    plt.show()


def plot_pattern(tx_array, w=None):
    dtheta = np.linspace(0, 2*np.pi, 360)
    dphi = np.linspace(0, np.pi, 360)
    theta, phi = np.meshgrid(dtheta, dphi)
    
    X = np.sin(theta)*np.cos(phi)
    Y = np.sin(theta)*np.sin(phi)
    Z = np.cos(theta)
    
    # So, I chose put antenna elements at x-y axis
    tx_array_x = 1
    #tx_array_x = int(np.sqrt(tx_array.size))
    tx_array_y = 1
    #tx_array_y = int(np.sqrt(tx_array.size))
    #tx_array_z = 1 
    tx_array_z = int(tx_array.size)
    
    wavelength = tx_array.wave_length
    d = tx_array.element_spacing
    k = 2 * np.pi * (1/wavelength)
    
    af = 0 
    for x in range(tx_array_x):
       for y in range(tx_array_y):
           for z in range(tx_array_z):
               delay = (x * X) + (y * Y) + (z * Z)
               if w is None:
                   af = af + (np.exp(1j * k * d * delay))
               else:
                   af = af + (w[z] * np.exp(1j * k * d * delay))
                   #af = af + (np.exp(1j * k * d * delay)) 
    
    af = np.abs(af)
    X = af * X
    Y = af * Y
    Z = af * Z
    
    # View it.
    if w is None:
        s = mlab.mesh(X, Y, Z)
    else:
        s = mlab.mesh(X, Y, Z)
        
    #h = plt.contourf(X, Y, Z)
    #plt.show()
    mlab.colorbar(orientation='vertical')
    mlab.axes()
    mlab.show()




def PlotPattern2(antenna_array):
    """
    ref: Balanis 2016, chp.06
    ref: https://www.mtt.org/wp-content/uploads/2019/01/beamform_mmw_antarr.pdf
    ref: http://www.waves.utoronto.ca/prof/svhum/ece422/notes/15-arrays2.pdf
    ref: https://www.ece.mcmaster.ca/faculty/nikolova/antenna_dload/current_lectures/L13_Arrays1.pdf
    """
    num_tx_elements = antenna_array.size
    #fc = 60 * (10 ** 9) #60 GHz
    #fc = 60 * (10 ** 9) #60 GHz
    wavelength = antenna_array.wave_length
    #element_spacing =  wavelength/2 #in meters
    element_spacing =  antenna_array.element_spacing #5 / (10 ** 3) #wavelength/2 #in meters
    k = 2 * np.pi * 1/wavelength

    theta = np.arange(0, 180, 1)

    array_vec = np.arange(num_tx_elements)

    w = np.ones(num_tx_elements, dtype=complex)
    alpha = np.random.choice(theta)
    #alpha = 90 #np.random.choice(theta)
    print ('alpha: ', alpha)
    count = 1
    for i in range(len(w)):
        #t = np.random.choice(theta)
        t = alpha * count
        count += 1
        w[i] = 1 * np.exp(-1j * np.deg2rad(t))
    print (w)
    print (np.abs(w))

    product = np.zeros(len(theta), dtype=complex)
    f_psi = np.zeros(len(theta), dtype=complex)
    psi = np.zeros(len(theta))
    for i in range(len(theta)):
        t = np.deg2rad(theta[i])
        p = k * element_spacing * np.cos(t)
        psi[i] = p
        array_factor = np.array([np.exp(1j * n * p) for n in array_vec])

        product[i] = np.matmul(w.T, array_factor)
        product[i] = product[i]/np.matmul(w.conj().T, w)

        af0 = np.sum(array_factor) 
        af1 = af0 * np.exp(1j * p)
        af2 = af0 - af1 
        af3 = af2/(1 - np.exp(1j * p))
        f_psi[i] = af3#/len(array_vec)


    fig = plt.figure()

    angles = np.angle(product)
    mag = np.abs(product)
    fig.add_subplot(311, projection='polar')
    plt.polar(psi, mag)

    fig.add_subplot(312)
    plt.plot(psi, np.abs(product), label='AF x w')
    plt.legend()

    fig.add_subplot(313)
    plt.plot(psi, np.abs(f_psi), label='f(psi)')

    plt.legend()
    plt.show()



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
        codebook_dict[cw_id] = np.matrix(np.array(codeword, dtype=complex)).T
        #codebook_dict[cw_id] = np.matrix(np.zeros(len(codebook_json.keys()), dtype=complex)).T
        #print (np.shape(codebook_dict[cw_id]))
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

def conf_interval(x, conf_level=0.90):

    alpha = 1 - conf_level/100
    sample_size = len(x)
    sample_dist = np.sqrt(np.sum((x - np.mean(x)) ** 2)/(sample_size-1))

    if (sample_dist == 0.0 and sample_size < 30):
        print ('sample size too small for normal dist')
        return 0

    ci_values = None
    sample_mean = np.mean(x)
    sem = sample_dist/np.sqrt(sample_size) # Standard Error of Mean 

    if (sample_dist == 1.0 or sample_size < 30):
	# using T-student distribution
        if sample_size < 30:
            print(f'Small sample size: {sample_size}. It should be used only when the population has a Normal distribution.');
        ci_values = st.t.interval(conf_level, df=len(x)-1, loc=sample_mean, scale=sem)
        print (f't-student: {ci_values}')
    else:
        # using normal distribution
        ci_values = st.norm.interval(conf_level, loc=sample_mean, scale=sem)
        print (f'normal: {ci_values}')
    return sample_mean, ci_values
#x = np.array([-13.7,  13.1,  -2.8,  -1.1,  -3. ,   5.6])
#x = np.array([1.5, 2.6, -1.8, 1.3, -0.5, 1.7, 2.4])
#x = np.array([3.1, 4.2, 2.8, 5.1, 2.8, 4.4, 5.6, 3.9, 3.9, 2.7, 4.1, 3.6, 3.1, 4.5, 3.8, 2.9, 3.4, 3.3, 2.8, 4.5, 4.9, 5.3, 1.9, 3.7, 3.2, 4.1, 5.1, 3.2, 3.9, 4.8, 5.9, 4.2])
#sample_mean, ci_values = conf_interval(x, 0.90)
#print (sample_mean)
#print (ci_values)
def plot_errorbar(values, label, title):
    num_cols, num_rows = values.shape
    x = np.arange(num_rows)
    print (x)
    print (x.shape)
    #y = values
    y_mean = np.mean(values, axis=0)
    print (y_mean.shape)
    print (y_mean)
    var = np.mean((values - y_mean) ** 2, axis=0)
    print (var.shape)
    std_dev = np.sqrt(var)
    print (std_dev)
    e = (y_mean + std_dev) - (y_mean - std_dev)
    print (e)
    print (e.shape)
    plt.errorbar(x, y_mean, e, label=label)
    plt.legend()
    plt.title(title)
    plt.ylabel(r'E($\sigma_i$)')
    plt.xlabel(r'i')
    #return plt
    #plt.show()
#values = np.random.rand(10000,4)
#print (values.shape)
#plot_errorbar(values)

def get_beamforming_vector_from_sample(h):
    u, s, vh = svd(h)
    f = np.matrix(vh.conj()[0,:]) # vetor linha: 1xN
    #f = vh.conj()[0,:] # vetor linha: 1xN
    #adjust to hardwaare limited
    #n_rows, n_cols = np.shape(f) 
    #f = (1/np.sqrt(n_cols)) * np.exp(1j * np.angle(f))
    return f

def throw_lloyd(parm):

    data = {}

    trial_id = parm['trial_id']
    data['trial_id'] = trial_id

    results_dir = parm['results_dir']
    data['results_dir'] = results_dir

    json_filename = str(results_dir) + f'/result_{trial_id}.json'

    initial_alphabet_opt = parm['initial_alphabet_opt']
    data['initial_alphabet_opt'] = initial_alphabet_opt

    distortion_measure_opt = parm['distortion_measure_opt']
    data['distortion_measure_opt'] = distortion_measure_opt

    max_num_of_interactions = parm['max_num_of_interactions']
    data['max_num_of_interactions'] = max_num_of_interactions

    trial_random_seed = parm['trial_random_seed'] # trial_random_seed
    data['trial_random_seed'] = float(trial_random_seed)

    samples = parm['channel_samples']
    data['channel_samples_filename'] = parm['channel_samples_filename']

    num_of_samples, rx_array_size, tx_array_size = np.shape(samples)
    data['rx_array_size'] = rx_array_size
    data['tx_array_size'] = tx_array_size
    data['num_of_samples'] = num_of_samples

    num_of_levels = parm['num_of_levels']
    data['num_of_levels'] = num_of_levels 

    #phase_shift_resolution = parm['phase_shift_resolution']
    #data['phase_shift_resolution'] = phase_shift_resolution
    # Starting lloyd with an specific initial alphabet opt
    
    #if initial_alphabet_opt == 'katsavounidis':
    #    training_samples = np.array([get_beamforming_vector_from_sample(samples[i]) for i in range(num_of_samples)])
    #    initial_codebook = katsavounidis_initial_codebook_2(training_samples, num_of_levels)
    #    print (f'initial_alphabet_opt: {initial_alphabet_opt} -- initial_codebook.shape: {np.shape(initial_codebook)}')
    #    codebook, sets, mean_distortion_by_round = lloyd(initial_alphabet_opt, training_samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)
    #    pass

    if initial_alphabet_opt == 'xiaoxiao':
        training_samples = np.array([get_beamforming_vector_from_sample(samples[i]) for i in range(num_of_samples)])
        initial_codebook = xiaoxiao_initial_codebook(training_samples, num_of_levels)
        print (f'initial_alphabet_opt: {initial_alphabet_opt} -- initial_codebook.shape: {np.shape(initial_codebook)}')
        codebook, sets, mean_distortion_by_round = lloyd(initial_alphabet_opt, training_samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)
        pass

    #elif initial_alphabet_opt == 'sa':
    #    training_samples = np.array([get_beamforming_vector_from_sample(samples[i]) for i in range(num_of_samples)])
    #    initial_codebook = ???
    #    print (f'initial_alphabet_opt: {initial_alphabet_opt} -- initial_codebook.shape: {np.shape(initial_codebook)}')
    #    codebook, sets, mean_distortion_by_round = lloyd(initial_alphabet_opt, training_samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)
    #    pass

    #elif initial_alphabet_opt == 'unitary_until_num_of_elements':
    #    training_samples = np.array([get_beamforming_vector_from_sample(samples[i]) for i in range(num_of_samples)])
    #    initial_codebook = ???
    #    print (f'initial_alphabet_opt: {initial_alphabet_opt} -- initial_codebook.shape: {np.shape(initial_codebook)}')
    #    codebook, sets, mean_distortion_by_round = lloyd(initial_alphabet_opt, training_samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)
    #    pass

    elif initial_alphabet_opt == 'random_from_samples':
        # convert sampes from nxn to 1xn beamforming vector
        training_samples = np.array([get_beamforming_vector_from_sample(samples[i]) for i in range(num_of_samples)])
        # get some vectors as initial codebook
        #initial_codebook = np.array([get_beamforming_vector_from_sample(samples[i]) for i in np.random.choice(len(samples), num_of_levels, replace=False)])
        initial_codebook = np.array([training_samples[i] for i in np.random.choice(len(training_samples), num_of_levels, replace=False)])
        print (f'initial_alphabet_opt: {initial_alphabet_opt} -- initial_codebook.shape: {np.shape(initial_codebook)}')
        codebook, sets, mean_distortion_by_round = lloyd(initial_alphabet_opt, training_samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)

    elif initial_alphabet_opt == 'random':

        training_samples = np.array([get_beamforming_vector_from_sample(samples[i]) for i in range(num_of_samples)])
        phase_shift_resolution = 8 # bits
        initial_codebook = get_quantized_random_cb(tx_array_size, num_of_levels, phase_shift_resolution)
        print (f'initial_alphabet_opt: {initial_alphabet_opt} -- initial_codebook.shape: {np.shape(initial_codebook)}')
        codebook, sets, mean_distortion_by_round = lloyd(initial_alphabet_opt, training_samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)

    else:
        raise ValueError(f'Initial alphabet opt must be one of available opts in \'profile.json\' file')

    #if initial_alphabet_opt == 'random_from_samples':
    #    # convert sampes from nxn to 1xn beamforming vector
    #    training_samples = np.array([get_beamforming_vector_from_sample(samples[i]) for i in range(num_of_samples)])
    #    # get some vectors as initial codebook
    #    #initial_codebook = np.array([get_beamforming_vector_from_sample(samples[i]) for i in np.random.choice(len(samples), num_of_levels, replace=False)])
    #    initial_codebook = np.array([training_samples[i] for i in np.random.choice(len(training_samples), num_of_levels, replace=False)])
    #    codebook, sets, mean_distortion_by_round = lloyd(initial_alphabet_opt, training_samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, initial_codebook)

    #beamforming_vectors = np.array(get_beamforming_vector_from_sample(codebook[i]) for i in range(num_of_levels))
    #beamforming_vectors = matrix2dict(beamforming_vectors)
    data['codebook'] = encode_codebook(codebook)
    data['sets'] = encode_sets(sets)
    data['mean_distortion_by_round'] = encode_mean_distortion(mean_distortion_by_round)

    with open(json_filename, "w") as write_file:
        json.dump(data, write_file, indent=4)
    return 0

def lloyd(initial_alphabet_opt, samples, num_of_levels, num_of_iteractions, distortion_measure, initial_codebook=None):
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

    elif initial_alphabet_opt == 'random_from_samples' or initial_alphabet_opt == 'random' or initial_alphabet_opt == 'sa' or initial_alphabet_opt == 'katsavounidis' or initial_alphabet_opt == 'xiaoxiao':
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
            #print (f'codebook shape: {np.shape(codebook)}')

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
                            new_cw = new_cw/norm(new_cw) #complex_average(sub_set_of_samples_matrix[start:end])
                        else:
                            raise ValueError(f'Error: no distortion measure option chosen')
                    else:
                        new_cw = complex_average(dict2matrix(sub_set_of_samples))
                        new_cw = new_cw/norm(new_cw)
                        #new_cw = num_of_levels * new_cw/norm(new_cw)
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
