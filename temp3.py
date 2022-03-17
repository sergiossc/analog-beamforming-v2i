import load_lib
import numpy as np
from numpy.linalg import matrix_rank, svd, norm, det
from utils import gen_hermitian_matrix, loschnmtx
import mimo.arrayconfig as arraycfg
 
if __name__ == '__main__':

    tx_array = arraycfg.tx_array
    rx_array = arraycfg.rx_array

    num_samples = 1000
    n = tx_array.size
    samples = []
    for _ in range(num_samples):
        #h = gen_hermitian_matrix(n)
        complex_gain = 1 * np.exp(1j * 0)
        theta_range = np.linspace(0, 2*np.pi, 100)
        aoa = np.random.choice(theta_range)
        aod = np.random.choice(theta_range)
        h = loschnmtx(complex_gain, rx_array, tx_array, aoa, aod)
        print (np.allclose(h, h.conj().T))
        n_row, n_col = h.shape
        h = h.T.reshape(n_row * n_col)
        samples.append(h)
        
    samples = np.array(samples)
    samples = samples.reshape(num_samples, n_row*n_col)
    
    print (samples.shape)
    
    cov_matrix = np.cov(samples.T)

    cov_matrix = n * cov_matrix/norm(cov_matrix)

    print (cov_matrix.shape)
    print (cov_matrix)
    print (np.abs(cov_matrix))
    print (matrix_rank(cov_matrix))
    print (f'det: {det(cov_matrix)}')

    u, s, vh = svd(cov_matrix)
    print (s ** 2)
    print (np.sum(s ** 2))
 
    
