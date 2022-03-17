from utils import array_to_angular_domain, angular_to_array_domain, gen_channel
import numpy as np
from numpy.linalg import norm

nr = 6
nt = 6
ch_array = gen_channel(nr, nt, 1)

ch_angular = array_to_angular_domain(ch_array)

ch_array_est = angular_to_array_domain(ch_angular)

print('max error real=',np.max(np.real(ch_array_est-ch_array)))

print('max error imag=',np.max(np.imag(ch_array_est-ch_array)))


print (f'norm of array: {norm(ch_array)}')
print (f'norm of array_est: {norm(ch_array_est)}')
print (f'norm of angular: {norm(ch_angular)}')
