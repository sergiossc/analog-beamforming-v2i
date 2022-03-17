import numpy as np
from utils import richscatteringchnmtx, squared_norm, norm
from numpy.linalg import svd
import matplotlib.pyplot as plt

#np.random.seed(1234)

n = 4
m = 4

#snr_db = np.arange(-10, 50, 0.01) 
#snr = 10 ** (snr_db/10)
rho = 10 # dB
snr = 10 ** (rho/10)

c_vec = []
sv_vec = []

n_trials = 20000

n_streams = np.min([n, m])
#n_trials = 1000

var = 1.0
h = richscatteringchnmtx(n, m, var)
h = h/norm(h) # turn h an unitary vector
u, s, vh = svd(h * h.conj()) # perform singular value decomposition

for n_trial in range(n_trials):

    #s = s ** 2
    s_rand = np.random.rand(n_streams)
    s_rand = s_rand/norm(s_rand)

    s_rand = s_rand ** 2
    sv_vec.append(s_rand)

    c = 0
    for si in s_rand:
        c = c + np.log2(1 + snr * si * 1/m)
    ##print (si)
    #sv_vec.append(s_rand)
    c_vec.append(c)


#sv_vec = np.array(sv_vec)
c_vec = np.array(c_vec)
max_c = np.argmax(c_vec)
print (max_c)
print (sv_vec[max_c])

plt.plot(c_vec, 'b*', label=f'c_vec')
#plt.plot(np.sum(sv_vec, axis=1), 'r*', label=f'sv_vec')
plt.legend()
plt.title(f'Capacity vs trials')
plt.xlabel(f'#trial')
plt.ylabel(f'Capacity(bps/Hz)')
plt.grid()
plt.show()
