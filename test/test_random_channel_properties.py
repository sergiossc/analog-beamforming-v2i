import numpy as np
from utils import richscatteringchnmtx, squared_norm, norm
from numpy.linalg import svd
import matplotlib.pyplot as plt

#np.random.seed(1234)

n = 4
m = 4


n_trials = 1000
sv = []

for trial in range(n_trials):

    var = 1.0
    h = richscatteringchnmtx(n, m, var)
    h = np.sqrt(n*m) * (h/norm(h)) # turn h an unitary vector
    u, s, vh = svd(h) # perform singular value decomposition
    #s = s ** 2
    sv.append(s)


#max_c = np.argmax(c_vec)
#print (max_c)
#print (sv_vec[max_c])

plt.plot(np.sum(sv, axis=1), label=f'[{n},{m}]')
#plt.plot(np.sum(sv_vec, axis=1), 'r*', label=f'sv_vec')
plt.legend()
plt.title(f'SV vs Channel realization')
plt.xlabel(f'SV sum')
plt.ylabel(f'#trial')
plt.grid()
plt.show()
