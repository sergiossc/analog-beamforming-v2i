import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

def get_spatial_signature(array, omega):
    n = array['n']
    d = array['d']

    e = (1/np.sqrt(n)) * np.matrix([np.exp(-1j * 2 * np.pi * d * i * omega) for i in range(n)])
    return e

theta = np.linspace(0, 2 * np.pi, num=500)
omega = np.linspace(-2, 2, num=500)
#omega = np.cos(theta)

fc = 60 * (10 ** 9)
#wavelength = c/fc
d = 0.5
nr = 4

#rx_array = {'fc': fc, 'wavelength': wavelength, 'd': d, 'n': nr}
rx_array = {'fc': fc, 'd': d, 'n': nr}

e_rx_0 = get_spatial_signature(rx_array, 0/(d*nr))
e_rx_1 = get_spatial_signature(rx_array, 1/(d*nr))
e_rx_2 = get_spatial_signature(rx_array, 2/(d*nr))
e_rx_3 = get_spatial_signature(rx_array, 3/(d*nr))

psi_0 = []
psi_1 = []
psi_2 = []
psi_3 = []

for o in omega:
    psi_0.append(np.abs(e_rx_0 * get_spatial_signature(rx_array, o).conj().T)[0,0])
    psi_1.append(np.abs(e_rx_1 * get_spatial_signature(rx_array, o).conj().T)[0,0])
    psi_2.append(np.abs(e_rx_2 * get_spatial_signature(rx_array, o).conj().T)[0,0])
    psi_3.append(np.abs(e_rx_3 * get_spatial_signature(rx_array, o).conj().T)[0,0])

fig = plt.figure()
fig.add_subplot(111, projection='polar')
#plt.plot(psiomega, np.abs(af)/array_size)

plt.plot(omega, np.array(psi_0), label='0/L')
plt.plot(omega, np.array(psi_1), label='1/L')
plt.plot(omega, np.array(psi_2), label='2/L')
plt.plot(omega, np.array(psi_3), label='3/L')
plt.legend()
#plt.savefig('beamforming.png')
plt.show()
