import numpy as np
from numpy.random.mtrand import sample
from utils import richscatteringchnmtx, squared_norm, norm, covariance_matrix
from numpy.linalg import svd
import matplotlib.pyplot as plt

#np.random.seed(1234)

n = 4
m = 4

n_trials = 100000
sv = []
samples = []

for trial in range(n_trials):

    var = 1.0
    h = richscatteringchnmtx(n, m, var)
    h = np.sqrt(n*m) * (h/norm(h)) # turn h an unitary vector
    print (norm(h))
    u, s, vh = svd(h) # perform singular value decomposition
    #s = s ** 2
    samples.append(h)
    sv.append(s)
samples = np.array(samples)

print (f'covariance_mtx: \n{covariance_matrix(samples)}')
print (f'abs(covariance_mtx): \n{np.abs(covariance_matrix(samples))}')
print (f'ang(covariance_mtx): \n{np.rad2deg(np.angle(covariance_matrix(samples)))}')


#max_c = np.argmax(c_vec)
#print (max_c)
#print (sv_vec[max_c])

plt.plot(np.sum(sv, axis=1), label=f'[{n},{m}]')
print (f'mean: {np.sum(np.sum(sv, axis=1))/n_trials}')
#plt.plot(np.sum(sv_vec, axis=1), 'r*', label=f'sv_vec')
plt.legend()
plt.title(f'SV vs Channel realization')
plt.xlabel(f'SV sum')
plt.ylabel(f'#trial')
plt.grid()
#plt.show()
