import numpy as np
from utils import richscatteringchnmtx, squared_norm, norm
from numpy.linalg import svd
import matplotlib.pyplot as plt

#np.random.seed(1234)

n = 4
m = 4

snr_db_vec = np.arange(-10, 50, 0.01) 
snr_vec = 10 ** (snr_db_vec/10)

c_siso = []
c_bf = []
c_mimo = [] 
c_miso = [] 
c_simo = []  
c_mu = []

var = 1.0
h = richscatteringchnmtx(n, m, var)
h = h/norm(h) # turn h an unitary vector
u, s, vh = svd(h) # perform singular value decomposition
s = s ** 2

for snr in snr_vec: 
    c_siso.append(np.log2(1 + snr))
    c_bf.append(np.log2(1 + snr * s[0]))  # the strongest eigenchannels
    c_mu.append(np.sum([np.log2(1 + snr * si * 1/len(s)) for si in s])) # Sum of all eigenchannels
    c_mimo.append(len(s) * np.log2(1 + snr))
    c_miso.append(np.log2(1 + len(s) * snr))
    c_simo.append(np.log2(1 + len(s) * snr))


print (f'{np.sum(s)}')
print (f'{s}')
print (f'{s[0]}')

#max_c = np.argmax(c_vec)
#print (max_c)
#print (sv_vec[max_c])

plt.plot(snr_db_vec, c_siso, label=f'siso')
plt.plot(snr_db_vec, c_bf, label=f'beamforming')
plt.plot(snr_db_vec, c_mimo, label=f'mimo [{n}, {m}]')
plt.plot(snr_db_vec, c_mu, label=f'multplex gain mimo [{n}, {m}]')
plt.plot(snr_db_vec, c_miso, label=f'miso [{n}, {1}]')
plt.plot(snr_db_vec, c_simo, label=f'simo [{1}, {m}]')
#plt.plot(np.sum(sv_vec, axis=1), 'r*', label=f'sv_vec')
plt.legend()
plt.title(f'Capacity vs SNR')
plt.xlabel(f'SNR(dB)')
plt.ylabel(f'Capacity(bps/Hz)')
plt.grid()
plt.show()
