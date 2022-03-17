import numpy as np
from scipy.constants import c
from numpy.linalg import svd
import matplotlib.pyplot as plt


class path:
    def __init__(self, id, aod, aoa, received_power):
        self.id = str(id)
        self.aod = np.deg2rad(aod)
        self.aoa = np.deg2rad(aoa)
        self.received_power = np.power(10,received_power/10) # convert dB in W
        self.at = []
       
    def to_string(self):
        print ('id: ', self.id)
        print ('aod: ', np.rad2deg(self.aod))
        print ('aoa: ', np.rad2deg(self.aoa))
        print ('received_power: ', self.received_power)

def array_factor(p, array_vec):
    factor = 1/len(array_vec) # no cado do uso pra estimar o canal considerando TX e RX, usar 1/sqrt(len(array_vec).
    af = np.array([np.exp(-1j *  n * p) for n in array_vec])
    #af = np.array([np.exp(-1j * n * p) for n in tx_array])
    return factor * af

fc = 60 * (10 ** 9)
wavelength = c/fc
d = wavelength/2
k = 2 * np.pi * 1/wavelength

num_tx = 2
tx_array = np.arange(num_tx)

num_rx = 1
rx_array = np.arange(num_rx)

theta = np.arange(1,180)

num_paths = 3
np.random.seed(444)
paths = []
for i in range(num_paths):
    aod = np.random.choice(theta)
    aoa = np.random.choice(theta)
    received_power = np.random.choice([-2, -1, 0, 1, 2], 1, replace=False)  # same power for all paths: 1W
    p = path(i, aod, aoa, received_power)
    p.to_string()
    paths.append(p)

#p1 = path(1, 30, 50, 3)
#p1.to_string()
#p2 = path(2, 150, 25, 7)
#p2.to_string()
#paths.append(p1)
#paths.append(p2)


path_res = path(1000, 0, 0, 0)
path_res.at = np.zeros(len(tx_array))

h = np.zeros((num_rx, num_tx))
received_power_list = [p.received_power for p in paths]
factor = np.sqrt(num_rx * num_tx) * np.linalg.norm(received_power_list)/np.sum(received_power_list)


for path in paths:
    p_t = k * d * np.cos(path.aod)
    at = array_factor(p_t, tx_array) #steeering vector
    p_r = k * d * np.cos(path.aoa) 
    ar = array_factor(p_r, rx_array) #response vector
    outer_product = np.outer(ar.conj().T, at)
    complex_gain = path.received_power * np.exp(-1j * np.deg2rad(np.random.choice(theta)))
    h = h + (complex_gain * outer_product)

    path_res.at = path_res.at + (complex_gain * at)
h = h * factor
path_res.at = path_res.at * factor

print ('h', h)
u, s, vh = svd(h)
print ('u', u)
print ('s', s)
print ('vh', vh)
print ('vh[0]', vh[0])
w1 = vh[0]
print ('w1', w1)
print ('sum(w1.abs):', np.sum(np.abs(w1)))
w3 = np.zeros((len(tx_array)), dtype=complex)
for i in range(len(w1)):
    w3[i] = 1 * np.exp(1j * np.angle(w1[i]))

print ('w3:', w3)
print ('w3.abs:', np.abs(w3))
    

w2 = path_res.at
print ('w2', w2)
print ('sum(w2.abs):', np.sum(np.abs(w2)))

psi = np.zeros(len(theta))
gain_w1 = np.zeros(len(theta), dtype=complex)
gain_w2 = np.zeros(len(theta), dtype=complex)
gain_w3 = np.zeros(len(theta), dtype=complex)
f_psi = np.zeros(len(theta))

for i in range(len(theta)):
    t = np.deg2rad(theta[i])
    p = k * d * np.cos(t)
    psi[i] = p

    af = np.array([np.exp(1j *  n * p) for n in tx_array])
     
    prod1 = np.matmul(w1.T, af)
    prod2 = np.matmul(w2.T, af)
    prod3 = np.matmul(w3.T, af)

    gain_w1[i] = 1/len(tx_array)*(np.abs(prod1)**2)/np.abs(np.matmul(w1.conj().T, w1))
    gain_w2[i] = 1/len(tx_array)*(np.abs(prod2)**2)/np.abs(np.matmul(w2.conj().T, w2))
    gain_w3[i] = 1/len(tx_array)*(np.abs(prod3)**2)/np.abs(np.matmul(w3.conj().T, w3))


fig = plt.figure()

fig.add_subplot(111)
plt.plot(theta, np.abs(gain_w1), label='svd puro (w1)')
plt.plot(theta, np.abs(gain_w3), label='w1 com amplitude 1 (w3)')
plt.plot(theta, np.abs(gain_w2), label='somatorio para Nr=1 (w2)')

plt.legend()
plt.show()
