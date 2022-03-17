import numpy as np
from numpy import matmul
from scipy.constants import c
import matplotlib.pyplot as plt
import sys

class path:
    def __init__(self, id, aod):
        self.id = id
        self.aod = np.deg2rad(aod)
        self.at = None

theta = np.arange(0,180)
paths = []
num_paths = 2
#my_seed = np.random.choice(np.arange(100000))
my_seed = 633
np.random.seed(my_seed)
print ('my_seed: \n', my_seed)
for n in range(num_paths):
    p = path('p'+str(n), np.random.choice(theta))
    paths.append(p)

num_tx_elements = 4
fc = 60 * (10 ** 9) #60 GHz
wavelength = c/fc #in meters
element_spacing = wavelength/2 #in meters
array_vec = np.arange(num_tx_elements)
k = 2 * np.pi * 1/wavelength
at_res = np.zeros(len(array_vec))

gains = [1, 1]

for p in paths:
    print('path_id: \n', p.id)
    aod = p.aod
    print ('aod: ', aod)
    at = np.array([np.exp(-1j * 2 * np.pi * n * element_spacing * (1/wavelength) * np.cos(aod)) for n in range(len(array_vec))])
    complex_gain = np.random.choice(gains) * np.exp(-1j * np.random.choice(theta))
    print ('abs.complex_gain: \n', np.abs(complex_gain))
    at_res = at_res + (at * complex_gain)
    p.at = at

p_res = path('p_res', 0)
p_res.at = at_res
paths.append(p_res)

for p in paths:
    at = p.at
    prod = np.zeros(len(theta), dtype=complex)
    for i in range(len(theta)):
        t = np.deg2rad(theta[i])
        af = np.array([np.exp(1j * 2 * np.pi * n * element_spacing * (1/wavelength) * np.cos(t)) for n in range(len(array_vec))])
        product = np.matmul(at.T, af)
        product = (np.abs(product) ** 2)/np.abs(np.matmul(at.conj().T, at))#print ('product: \n', product)
        prod[i] = product/len(array_vec)
    plt.plot(theta, prod, label=p.id)
    plt.legend()
#plt.savefig('steering.png')
plt.show()
