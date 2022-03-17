from utils import get_beamforming_vector_from_sample
import numpy as np
from numpy.linalg import norm
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    pass
    print ('get_beamforming_vector_from_sample  test')
    pathfile_samples = sys.argv[1]
    print (pathfile_samples)

    error_list = []
    samples = np.load(pathfile_samples)
    for sample in samples:
        print ('------------')
        f = get_beamforming_vector_from_sample(sample)
        print (norm(f))
        print ('shape of f ', np.shape(f))
        n_rows, n_cols = np.shape(f)
        print ('n_cols:', n_cols)
        print ('n_rows:', n_rows)
        f_a = (1/np.sqrt(n_cols)) * np.exp(1j * np.angle(f))
        f = np.array(f) #.reshape(n_cols)
        f_a = np.array(f_a) #.reshape(n_cols)
        print ('norm of f_a ', np.shape(f_a))
        #print (norm(f_a))
        error = np.angle(f) - np.angle(f_a) 
        error = np.sum(error ** 2)
        #error = error.reshape(n_rows, n_cols)
        #error = [i for i in error]
        #error = np.angle(f) - np.angle(f_a)
        #error = error.flatten()
        print ('shape of error ', np.shape(error))
        error_list.append(error)
    plt.plot(error_list)
    plt.show()
        
