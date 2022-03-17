#!/usr/bin/python3.8
import numpy as np
from numpy.linalg import matrix_rank, norm, det, svd
#from numpy.linalg import matrix_rank
import load_lib
import lib.mimo.arrayconfig as phased_array
from utils import loschnmtx, richscatteringchnmtx, covariance_matrix, plot_errorbar
import matplotlib.pyplot as plt

def cov_matrix(samples):
    samples = np.array(samples)
    n_samples, n_rows, n_cols = samples.shape
    prod_sum = np.zeros((n_rows*n_cols, n_rows * n_cols))
    for sample in samples:
        vec1 = sample.T.reshape(n_rows * n_cols,1)
        vec2 = sample.conj().T.reshape(n_rows * n_cols, 1)
        prod = vec1 * vec2.T
        prod_sum = prod_sum + prod #.append(prod)
        #pass
    return prod_sum/n_samples
    #return r
rx_array = phased_array.rx_array
tx_array = phased_array.tx_array

phase = 0
complex_gain = 1.0 * np.exp(1j * phase)

num_samples = 200
s_los_list = []
s_w_list = []
s_ricean_k0_list = []
s_ricean_k75_list = []
s_ricean_k150_list = []
s_ricean_k225_list = []

for _ in range(num_samples):    

    #my_seed = 678
    #np.random.seed(my_seed)
    theta_a = np.random.choice(360)
    theta_d = np.random.choice(360)
    aoa = np.deg2rad(theta_a)
    aod = np.deg2rad(theta_d)
    
    h_los = loschnmtx(complex_gain, rx_array, tx_array, aoa, aod)
    h_los = tx_array.size * h_los/norm(h_los)
    #h_los_list.append(h_los)
    u_los, s_los, vh_los = svd(h_los)
    s_los_list.append(s_los ** 2) # ** 2)

    h_w = richscatteringchnmtx(rx_array.size, tx_array.size, 1.0)
    h_w = tx_array.size * h_w/norm(h_w)
    #h_w_list.append(h_w)
    u_w, s_w, vh_w = svd(h_w)
    s_w_list.append(s_w ** 2)# ** 2)

    k = 0
    h_ricean_k0 = np.sqrt(k/(1+k)) * h_los + np.sqrt(1/(1+k)) * h_w
    h_ricean_k0 = tx_array.size * h_ricean_k0/norm(h_ricean_k0)
    u_ricean_k0, s_ricean_k0, vh_ricean_k0 = svd(h_ricean_k0)
    s_ricean_k0_list.append(s_ricean_k0 ** 2)# ** 2)
    
    k = 1.0
    h_ricean_k75 = np.sqrt(k/(1+k)) * h_los + np.sqrt(1/(1+k)) * h_w
    h_ricean_k75 = tx_array.size * h_ricean_k75/norm(h_ricean_k75)
    u_ricean_k75, s_ricean_k75, vh_ricean_k75 = svd(h_ricean_k75)
    s_ricean_k75_list.append(s_ricean_k75 ** 2)# ** 2)

    k = 5.0
    h_ricean_k150 = np.sqrt(k/(1+k)) * h_los + np.sqrt(1/(1+k)) * h_w
    h_ricean_k150 = tx_array.size * h_ricean_k150/norm(h_ricean_k150)
    u_ricean_k150, s_ricean_k150, vh_ricean_k150 = svd(h_ricean_k150)
    s_ricean_k150_list.append(s_ricean_k150 ** 2)# ** 2)

    k = 20.0
    h_ricean_k225 = np.sqrt(k/(1+k)) * h_los + np.sqrt(1/(1+k)) * h_w
    h_ricean_k225 = tx_array.size * h_ricean_k225/norm(h_ricean_k225)
    u_ricean_k225, s_ricean_k225, vh_ricean_k225 = svd(h_ricean_k225)
    s_ricean_k225_list.append(s_ricean_k225 ** 2)# ** 2)
    #print (f'h_los rank: {matrix_rank(h_los)}')
    #print (f'h_w rank: {matrix_rank(h_w)}')
    
    #k_values = np.arange(20)
    #h_matrix_rank = [] #np.zeros(len(k_values))
    #h_matrix_det = [] #np.zeros(len(k_values))
    #k_perf_los = []
    #k_perf_w = []
    #s_sum = []
    #print (k_values)
    
    #for n in range(len(k_values)):
    #    k = k_values[n]
    #    h = (np.sqrt(k/(1+k)) * h_los) + (np.sqrt(1/(1+k)) * h_w)
    #    h = tx_array.size * h/norm(h)
    #    u, s, vh = svd(h)
    #    s_sum.append(s[0] ** 2)
    #    #h_matrix_rank.append(matrix_rank(h))
    #    #h_matrix_det.append(det(h))
    #    #k_perf_los.append(np.sqrt(k/(1+k)))
    #    #k_perf_w.append(np.sqrt(1/(1+k)))


x = np.arange(tx_array.size) + 1

#plt.plot(x, np.mean(np.array(s_los_list), axis=0), '-o', label='s_los')
#plt.plot(x, np.mean(np.array(s_w_list), axis=0), '-o', label='s_w')
#plt.plot(x, np.mean(np.array(s_ricean_k0_list), axis=0), '--*', label='s_ricean k=0')
#plt.plot(x, np.mean(np.array(s_ricean_k75_list), axis=0), '--*', label='s_ricean k=1')
#plt.plot(x, np.mean(np.array(s_ricean_k150_list), axis=0), '--*', label='s_ricean k=5')
#plt.plot(x, np.mean(np.array(s_ricean_k225_list), axis=0), '--*', label='s_ricean k=20')


#plt.title('MIMO channel (n = m = 4), 200 samples')
#plt.xlabel('i')
#plt.ylabel(r'$\sigma_i$')
#plt.legend()
#plt.show()

plot_errorbar(np.array(s_los_list), 's_los')
plot_errorbar(np.array(s_w_list), 's_w')

plot_errorbar(np.array(s_ricean_k0_list), 's_ricean, k=0')
plot_errorbar(np.array(s_ricean_k75_list), 's_ricean, k=75')
plot_errorbar(np.array(s_ricean_k150_list), 's_ricean, k=150')
plot_errorbar(np.array(s_ricean_k225_list), 's_ricean, k=225')

plt.show()