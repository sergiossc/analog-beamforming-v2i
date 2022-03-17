import numpy as np
from numpy import matmul
from numpy.linalg import svd
from scipy.constants import c
import matplotlib.pyplot as plt
import sys

#class path:
#    def __init__(self, id, aod):
#        self.id = id
#        self.aod = np.deg2rad(aod)
#        self.at = None


class Path:
    def __init__(self, id, aod, aoa, received_power):
        self.id = str(id)
        self.aod = np.deg2rad(aod)
        self.aoa = np.deg2rad(aoa)
        self.received_power = np.power(10,received_power/10) # convert dB in W
        self.at = None
 
    def to_string(self):
        print ('id: ', self.id)
        print ('aod: ', np.rad2deg(self.aod))
        print ('aoa: ', np.rad2deg(self.aoa))
        print ('received_power: ', self.received_power)
 
#def array_factor(p, array_vec):
#    factor = 1/len(array_vec) # no cado do uso pra estimar o canal considerando TX e RX, usar 1/sqrt(len(array_vec).
#    af = np.array([np.exp(-1j *  n * p) for n in array_vec])
#    #af = np.array([np.exp(-1j * n * p) for n in tx_array])
#    return factor * af


theta = np.arange(0,180)
num_paths = 2
#my_seed = np.random.choice(np.arange(100000))
my_seed = 633
np.random.seed(my_seed)
print ('my_seed: \n', my_seed)
paths = []
for n in range(num_paths):
    aod = np.random.choice(theta)
    aoa = np.random.choice(theta)
    received_power = np.random.choice([0.000001, 0.01, 1.0])
    p = Path(n, aod, aoa, received_power)
    paths.append(p)

num_tx_elements = 4
num_rx_elements = 4

fc = 60 * (10 ** 9) #60 GHz
wavelength = c/fc #in meters
element_spacing = wavelength/2 #in meters
tx_array_vec = np.arange(num_tx_elements)
rx_array_vec = np.arange(num_rx_elements)
k = 2 * np.pi * 1/wavelength
at_res = np.zeros(len(tx_array_vec))


h = np.zeros((num_rx_elements, num_tx_elements))
received_power_list = [p.received_power for p in paths]
factor = np.sqrt(num_rx_elements * num_tx_elements) * np.linalg.norm(received_power_list)/np.sum(received_power_list)

for p in paths:

    print('path_id: \n', p.id)
    aod = p.aod
    aoa = p.aoa
    print ('aod: ', aod)
    print ('aoa: ', aoa)
    at = np.array([np.exp(-1j * 2 * np.pi * n * element_spacing * (1/wavelength) * np.cos(aod)) for n in range(len(tx_array_vec))])
    ar = np.array([np.exp(-1j * 2 * np.pi * n * element_spacing * (1/wavelength) * np.cos(aoa)) for n in range(len(rx_array_vec))])
    complex_gain = p.received_power * np.exp(-1j * np.deg2rad(np.random.choice(theta)))
    print ('abs.complex_gain: \n', np.abs(complex_gain))
    at_res = at_res + (at * complex_gain)
    p.at = at

    outer_product = np.outer(ar.conj().T, at)
    h = h + (complex_gain * outer_product)
at_res = at_res * factor
h = h * factor


p_res = Path('p_res', 0, 0, 0)
p_res.at = at_res
paths.append(p_res)

print ('h', h)
u, s, vh = svd(h)
print ('u', u)
print ('s', s)
print ('vh', vh)
print ('vh[0]', vh[0])
w1 = vh[0]
print ('w1', w1)
w2 = p_res.at
print ('w2', w2)
p_svd = Path('p_svd', 0, 0, 0)
p_svd.at = w1
paths.append(p_svd)



for p in paths:
    at = p.at
    prod = np.zeros(len(theta), dtype=complex)
    for i in range(len(theta)):
        t = np.deg2rad(theta[i])
        af = np.array([np.exp(1j * 2 * np.pi * n * element_spacing * (1/wavelength) * np.cos(t)) for n in range(len(tx_array_vec))])
        product = np.matmul(at.T, af)
        product = (np.abs(product) ** 2)/np.abs(np.matmul(at.conj().T, at))#print ('product: \n', product)
        prod[i] = product/len(tx_array_vec)
    plt.plot(theta, prod, label=p.id)
    plt.legend()
#plt.savefig('steering.png')
plt.show()
