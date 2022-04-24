import numpy as np
from numpy.linalg import norm, svd
import matplotlib.pyplot as plt

def channel(num_rx, num_tx, variance):
    pass
    h = np.sqrt(variance/2) * np.random.randn(num_rx, num_tx) + (np.random.randn(num_rx, num_tx) * 1j)
    return h

if __name__ == "__main__":
    samples = np.load("s002-training_set_4x4.npy")
    print (np.shape(samples))

 
#    pass
#    num_of_trials = 100
#    num_rx = 4
#    num_tx = 4
#    variance = 1.0
#
    norm_values = []
    eigen_values_v0 = []
    eigen_values_v1 = []
    eigen_values_v2 = []
    eigen_values_v3 = []
    bf_gain_values = []
#
#    for i in  range(num_of_trials):
    np.seed = 5667
    num_samples, nr, nt = np.shape(samples) 
    set_samples = np.random.choice(np.arange(num_samples), 100)

    for i in set_samples:
        ch = samples[i]
        #ch = np.sqrt(nr * nt) * ch/norm(ch)
#        pass
#        ch = channel(num_rx, num_tx, variance)
#        ch = np.sqrt(num_rx * num_tx) * ch/norm(ch)
#
        u, s, vh = svd(ch)

        eigen_v0 = s[0] ** 2
        eigen_v1 = s[1] ** 2
        eigen_v2 = s[2] ** 2
        eigen_v3 = s[3] ** 2

        eigen_values_v0.append(eigen_v0)
        eigen_values_v1.append(eigen_v1)
        eigen_values_v2.append(eigen_v2)
        eigen_values_v3.append(eigen_v3)
        
        #norm_values.append(np.sum(s ** 2))

        vh = np.matrix(vh)
        u = np.matrix(u)

        f = vh[0,:]
        f = f.conj().T
        w = u[:,0]

        #print (f'f.shape: {np.shape(f)}')
        #print (f'w.shape: {np.shape(w)}')

        #f = (1/np.sqrt(num_tx)) * np.exp(1j * np.angle(f))

        bf_gain = np.abs(w.conj().T * (ch * f)) ** 2
        bf_gain = bf_gain[0,0]
        #print (bf_gain)
        bf_gain_values.append(bf_gain)

        norm_v = norm(ch)
        norm_values.append(norm_v)
        pass
    plt.plot(norm_values, label='norm_v')
    plt.plot(eigen_values_v0, label='eigen_v0')
    plt.plot(eigen_values_v1, label='eigen_v1')
    plt.plot(eigen_values_v2, label='eigen_v2')
    plt.plot(eigen_values_v3, label='eigen_v3')
    plt.plot(bf_gain_values, label='bf gain values', linestyle='--')
    plt.legend(loc='best')
    plt.show()
