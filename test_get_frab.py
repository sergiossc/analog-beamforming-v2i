import numpy as np
from numpy.linalg import norm
from utils import get_frab, gen_channel

def to_string(x):
    
    print (norm(x))
    print (np.abs(x))
    print (np.rad2deg(np.angle(x)))

if __name__ == "__main__":
    pass
    n_row = 1
    n_col = 4
    x = gen_channel(n_row, n_col, 1) # num_row, num_col, variance
    x = x/norm(x)
    to_string (x)
    print ('original')
    x_0 = 1/np.sqrt(n_col) * np.exp(1j * np.angle(x))
    to_string (x_0)
    print ('2-bit')
    x = get_frab(x_0, 2)
    to_string (x)
    print ('10-bit')
    x = get_frab(x_0, 10)
    to_string (x)
    
