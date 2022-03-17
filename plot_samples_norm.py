import numpy as np
from numpy.linalg import svd, matrix_rank, eigh, eig, eigvals
from lib.vq.utils import norm, squared_norm
import matplotlib.pyplot as plt
from utils import richscatteringchnmtx, decode_codebook
import sys
import json

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

if __name__ == "__main__":
    #profile_pathfile = 'profile.json' 
    #result_pathfile = sys.argv[1]
    samples_npy_pathfile = sys.argv[1]

    #with open(result_pathfile) as result:
    #    data = result.read()
    #    d = json.loads(data)

    channels = np.load(samples_npy_pathfile)
    
    n_samples, n_rx, n_tx = np.shape(channels)

    ch_norm = []
    lambda_sum = []
    eig_val_sum = []
    trace_h = []
    ch_max_eigen_value_svd = []
    ch_max_eigen_value_eig = []

    random_seed = 1234
    np.random.seed(random_seed)
    num_of_samples = 100
    ch_id_list = np.random.choice(len(channels), num_of_samples, replace=False)

    for ch_id in ch_id_list:
        ch = channels[ch_id]
        #ch = ch/norm(ch)
        print (check_symmetric(ch))
        trace_h.append(np.trace(ch * ch.conj().T))
        ch_norm.append(squared_norm(ch))

        #u, s, v = svd(ch * ch.conj().T)
        u, s, v = svd(ch)
        s = s ** 2
        lambda_sum.append(np.sum(s))
        ch_max_eigen_value_svd.append(s[0])

        #eig_val, eig_vec = eigh(ch * ch.conj().T)
        eig_val = eigvals(ch * ch.conj().T)
        #eig_val = eigvals(ch)
        #eig_val = np.abs(eig_val)
        #eig_val = eig_val ** 2
        eig_val = np.sort(eig_val)
        eig_val_sum.append(np.sum(eig_val))
        print (np.shape(eig_val))
        print (f'eig_values:\n{eig_val}')
        print (f'type of eig_val:\n{type(eig_val)}')
        ch_max_eigen_value_eig.append(eig_val[-1])

    ch_norm = np.array(ch_norm)
    ch_max_eigen_value_svd = np.array(ch_max_eigen_value_svd)
    ch_max_eigen_value_eig = np.array(ch_max_eigen_value_eig)
    lambda_sum = np.array(lambda_sum)
    eig_val_sum = np.array(eig_val_sum)
    trace_h = np.array(trace_h)

    fig, ax = plt.subplots()
    ax.ticklabel_format(useOffset=False)
    ax.plot(ch_norm, label='squared_norm')
    ax.plot(lambda_sum, label='lambda_sum')
    ax.plot(eig_val_sum, label='eig_val_sum')
    ax.plot(trace_h, label='trace_hh')
    
    ax.plot(ch_max_eigen_value_svd, label='eig_svd')
    ax.plot(ch_max_eigen_value_eig, label='eig_eig')
    plt.legend()
    plt.show()


    #for r, mean_distortion in distortion_by_round.items():
    #    mean_distortion = dict2matrix(mean_distortion)
    #    ax.plot(mean_distortion)
    #plt.ylabel('distortion')
    #plt.xlabel('# iterations')
    #plt.show()

 
