import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from scipy.constants import c
from mpl_toolkits import mplot3d
from numpy.linalg import norm


class PartitionedArray:
    def __init__(self, size, d, num_rf, wavelength, type_of="ULA"):
        self.size = num_tx
        self.d = d
        self.num_rf = num_rf
        self.wavelength = wavelength
        self.type_of = type_of
        self.ula = np.arange((size))

fc = 60 * (10 ** 9)
wavelength = c/fc
d = wavelength/2
k = 2 * np.pi * 1/wavelength

# Uniform Linear Array Setup
num_tx = 9
num_tx_rf = 4
tx_array = PartitionedArray(num_tx, d, num_tx_rf, wavelength, "ULA") # configuracao da antena de transmissao

length = 100 # of samples
theta = np.linspace(0, np.pi, length) # Elevation angles
phi = np.linspace(0, 2*np.pi, length) # Azimuth angles


u = np.sin(theta) * np.cos(phi)
v = np.sin(theta) * np.sin(phi)
dvec_u = np.array([np.exp(-1j * k * d * u[n] * np.arange(np.sqrt(tx_array.size))) for n in range(length)])
dvec_v = np.array([np.exp(-1j * k * d * v[n] * np.arange(np.sqrt(tx_array.size))) for n in range(length)])
print (dvec_u.shape)
print (dvec_v.shape)
af = np.zeros((length), dtype=complex)
for n in range(length):
    #print ('[n]: ', n)
    #print ('dvec_u[n]: ', dvec_u[n])
    #print ('dvec_v[n]: ', dvec_v[n])
    #print ('outer(*): ', np.outer(dvec_u[n], dvec_v[n])) 
    #print ('sum[outer(*)]: ', np.sum(np.outer(dvec_u[n], dvec_v[n]))) 
    af[n] = np.sum(np.outer(dvec_u[n], dvec_v[n])) 
    print ('af[n]: \n', af[n]) 

#af = np.outer(dvec_y, dvec_x)
#print (af.shape)
#af = dvec_x
#
#abs_af = np.zeros((length))
##psi = np.zeros((length))
#psi = k * d * np.cos(theta)
#f_psi = np.zeros((length))
##txomega_y = np.zeros((length))
#
#for n in range(length):
#
#    af = np.exp(-1j * psi[n] * tx_array.ula)  # considering squared ULA where number of elements on x_axis is equal y_axis...
#    af1 = np.sum(af)
#    af2 = af1 * np.exp(1j * psi[n])
#    af3 = af1 - af2
#    af4 = af3/(1 - np.exp(1j * psi[n]))
#    af5 = np.abs(af4)/len(tx_array.ula)
#    f_psi[n] = af5
#    #abs_af[n] = np.abs(af) #/np.matmul(af.conj().T, af))
#
#
X, Y = np.meshgrid(u, v)
#X, Y = np.meshgrid(psi, phi)
###Z = f(X, Y)
Z = np.array(np.meshgrid(np.abs(af))) #f(X, Y)
#
fig = plt.figure()
ax = plt.axes(projection='3d')
#
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title('surface');
#
ax.set_xlabel('psi')
ax.set_ylabel('phi')
ax.set_zlabel('gain')
#
#plt.savefig('my3dplot.png')
plt.show()
