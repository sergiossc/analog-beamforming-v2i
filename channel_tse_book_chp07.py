import numpy as np
from numpy.linalg import svd, norm
from scipy.constants import c

nt = 4
nr = 4

fc = 60 * (10 ** 9)
lambda_c = c/fc

tx_array = np.arange(nt)
rx_array = np.arange(nr)

aod = np.pi/4
omega_t = np.cos(aod)
delta_t = 0.5 * lambda_c
et_omega = 1/np.sqrt(nt) * np.array([np.exp(-1j * i * 2 * np.pi * delta_t * omega_t) for i in tx_array])
et_omega = np.matrix(et_omega)

#aoa = np.deg2rad(45)
aoa = -np.pi/4
omega_r = np.cos(aoa)
delta_r = 0.5 * lambda_c
er_omega = 1/np.sqrt(nr) * np.array([np.exp(-1j * i * 2 * np.pi * delta_r * omega_r) for i in rx_array])
er_omega = np.matrix(er_omega)

a = 1.0
d = np.sqrt(2) * 10
channel = a * np.sqrt(nr * nt) * np.exp(-1j * 2 * np.pi * d * (1/lambda_c) * 0)  * (er_omega.T * et_omega.conj())
#channel = a * np.sqrt(nt) * np.exp(1j * 0)  * (et_omega)
channel = nt * channel/norm(channel)
print (channel)
print (norm(channel))


# channel decomposition
u, s, vh = svd(channel)
u = np.matrix(u)
vh = np.matrix(vh)
precoder = vh.conj()[:, 0]
combining = u.conj()[:, 0]

# channel recover
# y = combining * H * precoder * x + combining * noise
# I1m interesting on product | combining * H * precoder|
print (precoder)
print (combining)
