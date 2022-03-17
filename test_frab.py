from utils import get_frab
from numpy.linalg import norm
import numpy as np
import sys

if __name__ == "__main__":

    nrows = int(sys.argv[1])
    ncols = int(sys.argv[2])
    b = int(sys.argv[3])

    complex_mat = np.matrix(np.random.rand(nrows, ncols) + 1j * np.random.rand(nrows, ncols))   

    #for col in range(ncols):
    #    column = complex_mat[:,col]
    #    complex_mat[:,col] = column/norm(column)

    #print (f'\ncomplex vec NORMALIZED: \n{complex_mat}')

    #for col in range(ncols):
    #    print (norm(complex_mat[:,col]))


    complex_mat_frab = get_frab(complex_mat, b) 

    print (complex_mat)
    print (f'\n')
    print (complex_mat_frab)
    print (f'\n')
    complex_mat_diff = complex_mat - complex_mat_frab
    print (complex_mat_diff.conj().T * complex_mat_diff)
