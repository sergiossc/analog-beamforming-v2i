import numpy as np
from numpy.linalg import svd, matrix_rank
from lib.vq.utils import norm
import matplotlib.pyplot as plt
from utils import * #richscatteringchnmtx, decode_codebook
import sys
import json
import pandas as pd
import os

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

def beamsweeping(ch, cb_dict):

    p_est_max = -np.Inf
    cw_id_max_tx = ''
    cw_id_max_rx = ''
    n_rx, n_tx = np.shape(ch)
    
    for k in cb_dict.keys():
        cw_tx = cb_dict[k]
        cw_tx = np.matrix(cw_tx)
        print (f'shape of cw_tx: {np.shape(cw_tx)}')
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
            print (f'shape of cw_rx: {np.shape(cw_rx)}')
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
            p_s = np.abs(p_s.conj() * p_s)
            #p_s = np.abs(p_s.conj() * p_s)
            if p_s > p_est_max:
                p_est_max = p_s
                cw_id_max_tx = k
                cw_id_max_rx = l

    return p_est_max, cw_id_max_tx, cw_id_max_rx

if __name__ == "__main__":


    profile_pathfile = 'profile-rt.json' 
    with open(profile_pathfile) as profile:
        data = profile.read()
        d = json.loads(data)

    prefix_pathfiles = d['results_directory']
    pathfile_test_samples = d['test_channel_samples_files']
    result_files = os.listdir(prefix_pathfiles)
    pathfiles = check_files(prefix_pathfiles, result_files)
    print ('# of json files: ', len(pathfiles))

    # PLOT CHART EFFORT
    #gridsize = (3, 2)
    #fig = plt.figure(figsize=(12, 8))
    #x, y = gridsize
    #pos_axes = [(i, j) for i in range(x) for j in range(y)]
    #axes = []
    #for pos in pos_axes:
    #    ax = plt.subplot2grid(gridsize, pos)
    #    axes.append(ax)
    #p_axes_count = 0
    
    df = pd.DataFrame() 
 
    for pathfile_id, pathfile in pathfiles.items():

        with open(pathfile) as result:
            data = result.read()
            d = json.loads(data)


        result_pathfile = pathfile #sys.argv[1]
    
        
        #channels = np.load('/home/snow/github/land/dataset/npy_files_s007/training_set_4x4.npy')
        initial_alphabet_opt = d['initial_alphabet_opt']
        title = f'{result_pathfile}, ' + initial_alphabet_opt
        print (f'{title}')
        print (f'*********************{pathfile_test_samples}') 
        #channels = np.load('s002-test_set_4x4.npy')
        channels = np.load(pathfile_test_samples)
        #my_cb = np.load('cb_s002_rx4_nlos_4x4.npy')
        codebook_json = d['codebook']
        cb_dict = decode_codebook(codebook_json)
    
        new_cb_dict = {}
        count = 0
        for k, v in cb_dict.items():
            count += 1
            new_cb_dict[count] = v
    
        cb_dict = new_cb_dict
    
        cb_keys = cb_dict.keys()
        #cb_hist = [(k, l) for k in cb_keys.keys() for l in cb_keys.keys()]
        cb_hist = {cw_id_pairwise: 0 for cw_id_pairwise in [(k, l) for k in cb_keys for l in cb_keys]}
        print (f'cb_keys: {cb_keys}')
        print (f'cb_hist: {cb_hist}')
        
        print (f'shape of channels: {np.shape(channels)}')
        n_samples, n_rx, n_tx = np.shape(channels)
    
        #num_of_trials = n_samples
        num_of_trials = 50
        np.random.seed(1234)
        ch_id_list = np.random.choice(len(channels), num_of_trials, replace=False)
        
        p_real = []
        p_est = []
    
        for ch_id in ch_id_list:
            ch = np.matrix(channels[ch_id])
            n = np.shape(ch)[0]
            m = np.shape(ch)[1]
            ch = ch/norm(ch)
            ch = np.sqrt(n*m) * ch
            #ch = a/norm(a)
            
            u, s, vh = svd(ch * ch.conj().T)    # singular values 
            #s = s ** 2 #eigenvalues
            print ('eigenvalues of channel----')
            print (s)
            print (np.sum(s))
            print ('----')
            #s = s ** 2 #eigenvalues
            vh = np.matrix(vh) 
            u = np.matrix(u) 
            f = vh[0,:]
            w = u[:,0]
        
            p = w.conj().T * (ch * f.conj().T)
            #p = np.abs(p)
            p = np.abs(p.conj() * p)
        
            p_real.append(p)
            
            #prod = ch.conj().T * ch
            
            p_est_max, cw_id_tx, cw_id_rx = beamsweeping(ch, cb_dict)
            p_est.append(p_est_max)
            cb_hist[(cw_id_tx,cw_id_rx)] += 1
        #p_real = np.array(p_real).reshape(num_of_trials)
        p_real = np.array(p_real)
        print (f'mean of real: {np.mean(p_real)}')
        #p_est = np.array(p_est).reshape(num_of_trials)
        p_est = np.array(p_est)
        
        print (f'mean of estmated: {np.mean(p_est)}')
        #print (f'cb_hist: {cb_hist}')
        #print (f'len(cb_hist): {len(cb_hist)}')
        ##df = pd.DataFrame(cb_hist.values(), index=cb_hist.keys(), columns=[f'{initial_alphabet_opt}'])
        #df[f'{initial_alphabet_opt}'] = cb_hist.values()
        #df.index = cb_hist.keys()
        #print (f'df: {df}')
    #df.plot.bar(subplots=True, legend=True, title=['','','','','',''], layout=(3,2), sharey=True, sharex=False, figsize=(24, 16), fontsize=8, xlabel='codeword pairwise', ylabel='# of samples', grid=True)
    #plt.show()
