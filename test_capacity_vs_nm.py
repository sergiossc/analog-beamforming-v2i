import numpy as np
from utils import richscatteringchnmtx, squared_norm, norm
from numpy.linalg import svd
import matplotlib.pyplot as plt

n = 4
m = 4

#nm_set = [[4, 4], [16, 16], [64, 64], [128, 128]]
nm_set = [[1, 1], [4, 10], [10, 4]]

snr_db = np.arange(-10, 50, 0.01) 
snr = 10 ** (snr_db/10)

c_vec = []
sv_vec = []


for nm in nm_set:

    n = nm[0]
    m = nm[1]

    #n_streams = np.min([n, m])
    #n_trials = 1000

    var = 1.0
    h = richscatteringchnmtx(n, m, var)
    h = h/norm(h) # turn h an unitary vector
    u, s, vh = svd(h) # perform singular value decomposition

    #s = s ** 2
    sv_vec.append(np.sum(s))

    c = np.zeros(len(snr))
    for si in s:
        c = c + np.log2(1 + snr * si/m)
    #print (si)
    c_vec.append(c)


c_vec = np.array(c_vec)
sv_vec = np.array(sv_vec)

print (np.shape(c_vec[0]))

for nm_item in range(len(nm_set)):
    pass
    #print (f'nm_item: {nm_item}')
    #plt.plot(snr_db, c_vec[nm_item], label=f'n,m: {nm_set[nm_item]}')
plt.plot(sv_vec, label=f'sum of sv_vec')
plt.legend()
plt.title(f'C')
plt.xlabel(f'SNR(dB)')
plt.ylabel(f'Capacity(bps/Hz)')
plt.grid()
plt.show()
