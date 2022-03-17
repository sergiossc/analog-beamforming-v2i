from utils import fftmatrix
import numpy as np

fftmat, ifftmat = fftmatrix(4, 2)

print (np.shape(fftmat))
