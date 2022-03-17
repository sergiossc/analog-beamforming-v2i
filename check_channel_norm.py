from scipy.io import loadmat
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import sys

npy_filename = sys.argv[1]

channel_data = np.load(npy_filename)

#mimo_fixed_data = loadmat('mimo_fixed.mat')

#for k, v in mimo_fixed_data.items():
#    print (k)


#ch_set = mimo_fixed_data['Harray']
#ch_set = mimo_fixed_data['Hvirtual']

norm_v = []

for ch in channel_data:
    print (np.shape(ch))
    print (norm(ch))
    norm_v.append(norm(ch))
    print (f'\n---')
channel_data = np.array(channel_data)
print (np.shape(channel_data))
#samples = np.save('mimo_fixed_v.npy', ch_set)
plt.plot(norm_v)
plt.show()
    
    
