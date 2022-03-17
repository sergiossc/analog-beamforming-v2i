import sys
import uuid
import numpy as np
from numpy.linalg import svd
from ratecalculations import get_rate

def dft_codebook(tx_array, Q=1):
    print ('Generating DFT codebook...')
    K = Q * tx_array.size
 
    seq = np.matrix(np.arange(K))
    mat = seq.conj().T * seq
    cb = np.exp(1j * 2 * np.pi * mat / K)
    codebook = {}
    for cw in cb:
        cw_id = uuid.uuid4()
        cw = cw[:,0:int(tx_array.size)]
        codebook[cw_id] = cw 
    print ('Done.')
    print ('Number of codewords: ', len(codebook))

    cb_id = uuid.uuid4()
    #filename = 'dftcodebook_' + str(cb_id) + '.npy'
    filename = 'dftcodebook.npy'
    np.save(filename, codebook)
    print('Codebook saved in ', filename) 

    return codebook

def codeword_by_path(tx_array):
    theta_deg = 30
    phi_deg = 45

    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    
    wavelength = tx_array.wave_length
    d = tx_array.element_spacing
    k = 2 * np.pi * (1/wavelength)
    u = k * d * np.sin(theta_rad) * np.sin(phi_rad)
    v = k * d * np.sin(theta_rad) * np.cos(phi_rad)
    delay = u + v
    w = np.arange(tx_array.size) * delay
    w = np.exp(1j * w)
    return w


def lloyd_codebook(tx_array, codebook_sample_filename):
    #codebook_sample_filename = '31d60ce5-838c-4588-8641-44a2133d9dfc.npy'
    est_channel = np.load(codebook_sample_filename,allow_pickle='TRUE').item()
    est_codebook = {}
    for est_sample_id, est_sample in est_channel.items():
        u, s, vh = svd(est_sample)
        w = vh[0]
        cw = np.zeros((1,tx_array.size), dtype=complex)
        for i in range(tx_array.size):
            cw[0,i] = 1 * np.exp(1j * np.angle(w[i]))
        est_codebook[est_sample_id] = cw
    return est_codebook


def beam_selection(sample, codebook, snr=1):
    # Doing selection beam...
    max_rate = -100.0
    max_cw_id = ''
    for cw_id, cw in codebook.items():

        r = get_rate(sample, cw, snr)
        if r > max_rate:
            max_rate = r
            max_cw_id = cw_id
    return max_cw_id, max_rate

def get_opt_cw_from_channel_sample(tx_array, sample):
    h = sample
    u, s, vh = svd(h)
    w = vh[0]
    #print ('w: ', w.shape)
    cw = np.zeros((tx_array.size), dtype=complex)
    for i in range(tx_array.size):
        cw[i] = 1 * np.exp(1j * np.angle(w[0,i]))
 
    return cw
