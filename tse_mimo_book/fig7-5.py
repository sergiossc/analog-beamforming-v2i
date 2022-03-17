import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

def spatial_signature(phased_array, omega):
    L = phased_array['L']
    n = phased_array['n']
    d = L/n
    form_factory = phased_array['form_factory']
    fc = phased_array['fc']
    wavelength = c/fc
    e = (1/np.sqrt(n)) * np.matrix([np.exp(-1j * 2 * np.pi * d/wavelength * omega * i) for i in range(phased_array['n'])])
    return e

if __name__ == "__main__":
    theta = np.linspace(-np.pi, np.pi, num=500)
    rx_array = {'fc': 60*(10**9), 'n': 4, 'form_factory': 'ULA', 'L': 8}
    tx_array = {'fc': 60*(10**9), 'n': 4, 'form_factory': 'ULA', 'L': 8}

    L = rx_array['L']
    n = rx_array['n']
    d = L/n

    #omega = np.cos(theta)
    omega = np.linspace(-2, 2, num=1000)
    psi = 2 * np.pi * d * omega

    e_rx_omega_0 = spatial_signature(rx_array, 0)
    f_omega = []

    for value in omega:
        ##f = np.abs(e_rx_omega_0.conj() * spatial_signature(rx_array, value).T)
        ##f_omega.append(f[0,0])
        ##print (type(f))
        ##print (f[0,0])
        af = (1 - np.exp(1j * n * 2 * np.pi * d * value))/(1 - np.exp(1j * 2 * np.pi * d * value))
        f_omega.append(np.abs(af))
    plt.plot(omega, np.array(f_omega))
    plt.show()