import numpy as np
import matplotlib.pyplot as plt
from utils import richscatteringchnmtx, norm, squared_norm
from numpy.linalg import svd

snr_db = np.arange(0, 50, 0.01)
snr = 10 ** (snr_db/10)

#SISO capacity
#c_siso = np.log2(1 + snr)
#plt.plot(snr_db, c_siso)

#MIMO capacity
n = 4
N = n
M = n

var = 1.0
h = richscatteringchnmtx(N, M, var)
h = n * h/norm(h)

print(f'shape of sample: {np.shape(h)}')
print(f'norm of sample: {norm(h)}')

#u, s, vh = svd(h.T * h.conj())
u, s, vh = svd(h)
f = vh.conj()
g = u.conj()

new_h = np.dot(g, np.dot(h, f))
new_u, new_s, new_vh = svd(new_h)
s = s ** 2
new_s = new_s ** 2
print (f'sum of s: {np.sum(s)}')
print (f's: {s}')
print (f'sum of new_s: {np.sum(new_s)}')
print (f'new_s: {new_s}')

gain_list = []

for i in range(len(s)):
    print ('--------->\ni: {i}')
    f0 = np.matrix(f[:, i])
    g0 = np.matrix(g[:, i])

    print(f'shape of f0: {np.shape(f0)}')
    #print(f'norm of f0: {norm(f0)}')

    print(f'shape of g0: {np.shape(g0)}')
    #print(f'norm of g0: {norm(g0)}')
    #print (np.allclose(h, np.dot(u[:,:] * s, vh)))

    #prod_01 = np.dot(h, f_0.T)
    prod0 = np.dot(g0, np.dot(h, f0.T))
    print(f'**shape of prod0: {np.shape(prod0)}')
    print(f'**norm of prod0: {norm(prod0)}')
    print(f'**lambda_{i}: {s[i]}')
    gain = norm(prod0)
    gain_list.append(gain)
print (f'sum of s: {np.sum(s)}')
print (f'sum of gain: {np.sum(gain_list)}')
gain_list = np.array(gain_list)
plt.plot(gain_list, 'b*')
plt.plot(new_s, 'go')
plt.plot(s, 'r*')
plt.show()
##prod1 = np.dot(h, f)
##prod = np.dot(g, prod1)
#
##print(f'shape of prod: {np.shape(prod)}')
##print(f'norm of prod: {norm(prod)}')
#
##print (np.allclose(h, prod))
##c_mimo = np.zeros(len(snr))
##c_mimo1 = np.zeros(len(snr))
##
##f = vh.conj()
#
##g = u.conj()
##prod1 = h.conj() * h #g * np.dot(h, f)
##prod2 = u * s * np.dot(s.T * c.conj()) #g * np.dot(h, f)
##gamma1 = np.sum(prod1) #g * np.dot(h, f)
###gamma2 = g * np.dot(h, f)
##print (f'gamma1: {gamma1}')
###c_mimo1 = c_mimo1 + np.log2(1 + i * snr)
