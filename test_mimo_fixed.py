from scipy.io import loadmat
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt

mimo_fixed_data = loadmat('mimo_fixed.mat')

for k, v in mimo_fixed_data.items():
    print (k)


ch_set = mimo_fixed_data['Harray']
#ch_set = mimo_fixed_data['Hvirtual']

norm_v = []

for ch in ch_set:
    print (np.shape(ch))
    print (norm(ch))
    norm_v.append(norm(ch))
    print (f'\n---')
ch_set = np.array(ch_set)
print (np.shape(ch_set))
samples = np.save('mimo_fixed_v.npy', ch_set)
plt.plot(norm_v)
plt.show()
    
    
