import numpy as np
import load_lib
from utils import *
from numpy.linalg import matrix_rank, svd
import matplotlib.pyplot as plt

def beam_selection(codebook, sample):
    effective_channel_gain = 0
    for codeword in codebook:
        effective_channel_gain = matmul(sample, codeword)

if __name__ == '__main__':
    
    npy_validation_files = [
    			r'/home/snow/github/land/dataset/npy_files_s007/validation_set_4x4.npy',
    			r'/home/snow/github/land/dataset/npy_files_s007/validation_set_4x16.npy',
    			r'/home/snow/github/land/dataset/npy_files_s007/validation_set_4x64.npy',
    			r'/home/snow/github/land/dataset/npy_files_s007/validation_set_16x16.npy',
    			r'/home/snow/github/land/dataset/npy_files_s007/validation_set_64x64.npy'
    			]
    
    npy_files_gain = []
    for npy_file in npy_validation_files:
        print (f'{npy_file}')
        samples = np.load(npy_file)
        num_samples, num_rx, num_tx = np.shape(samples)
        print (f'# of samples: {num_samples}')
        gain_samples = []
        gain_samples2 = []
        for sample in samples:
            r = matrix_rank(sample)
            u, s, vh = svd(sample.T, full_matrices=True)

            f = vh.conj()[:,0]
            g = u.conj()[:,0]
            opt = np.dot(g.conj().T, np.dot(sample.conj().T, f))
            #opt = np.dot(g.T, np.dot(sample.T, f))
            opt = np.matrix(opt)

            n = int(s.shape[0])
            sample_rec =  np.matrix(np.dot(u[:, :n] * s, vh))
            sample_rec = sample_rec.T
            
            #f = np.matrix(vh[:,0])
            #g = np.matrix(u[:,0])
            #v1 = sample.conj() * f.T
            #v2 = g.conj() * v1
            #value = norm(v2)
            value1 = np.abs(squared_norm(sample))
            value2 = np.abs(squared_norm(sample_rec)) #np.abs(norm(sample_rec))
            value = squared_norm(opt)
            #value = norm(np.matrix(g))
            gain_samples.append(value)
        npy_files_gain.append(gain_samples)

    npy_files_gain = np.array(npy_files_gain)
    print (f'npy_files_gain.shape: {np.shape(npy_files_gain)}') 


    #plt.plot(npy_files_gain[0], label='4x4')
    #plt.plot(npy_files_gain[1], label='4x16')
    #plt.plot(npy_files_gain[2], label='4x64')
    #plt.plot(npy_files_gain[3], label='16x16')
    plt.plot(npy_files_gain[4], label='64x64')
    plt.legend()
    plt.show()
