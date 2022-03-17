'''
https://www.mathworks.com/matlabcentral/fileexchange/27932-3d-array-factor-of-a-4x4-planar-array-antenna
'''
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from scipy.constants import c
from mpl_toolkits import mplot3d
from numpy.linalg import norm

class _PartitionedArray:
    def __init__(self, size, d, num_rf, wavelength, type_of="ULA"):
        self.size = size
        self.d = d
        self.num_rf = num_rf
        self.wavelength = wavelength
        self.type_of = type_of
        self.ula = np.arange((size))

def patter3d():
    fc = 60 * (10 ** 9)
    wavelength = c/fc
    d = wavelength/2
    k = 2 * np.pi * 1/wavelength
    
    # Uniform Planar Array Setup
    num_tx = 16
    num_tx_rf = 6
    tx_array = _PartitionedArray(num_tx, d, num_tx_rf, wavelength, "UPA") # configuracao da antena de transmissao
    
    length = 200  # of samples
    theta0 = np.linspace(0, np.pi, length) # Elevation angles
    phi0 = np.linspace(0, 2*np.pi, length) # Azimuth angles
    
    #theta, phi = np.meshgrid(theta0, phi0)
    phi, theta = np.meshgrid(phi0, theta0)
    
    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)
    
    N = int(np.sqrt(tx_array.size))
    M = int(np.sqrt(tx_array.size))
    
    ab = np.ones((N, M))
    #ab_phase = np.arange(N*M).reshape(N,M)  * (30*np.pi/180)
    #ab_phase = np.array([[0, 10, 20],[0, 10, 20],[0, 10, 20]]) * (np.pi/180)
    ab_phase = (np.pi/2) + (np.random.randn(N,M) * np.pi/2)  # random phases: mean pi/2, from 0 until pi
    #ab_phase = np.array([[0., 0.00013297, 0.00026595, 0.00039892], [0.00521822, 0.00535119, 0.00548416, 0.00561714], [0.01043643, 0.0105694, 0.01070238, 0.01083535], [0.01565465, 0.01578762, 0.01592059, 0.01605357]])
    #ab_coe = ab * np.exp(1j * ab_phase)
    ab_coe = np.random.rand(N,M) + np.random.rand(N,M) * 1j
    
    af = 0
    for n in range(N):
        for m in range(M):
            af = ab_coe[n][m] * np.exp(-1j * k * d * n * u) * np.exp(-1j * k * d * m * v) + af
            #af = np.exp(-1j * k * d * n * u) * np.exp(-1j * k * d * m * v) + af
            
    
    print ('af.shape: ', af.shape)
    print ('af: ', af)
    
    total_power = np.sum(ab ** 2)
    af_power = af ** 2
    normalized_af = np.abs(af_power/total_power) * 1/num_tx
    
    #af = af ** 2
    print ('af: ', af)
    af = np.abs(af)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(u, v, af, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface');
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('af')
    #plt.savefig('my3dplot.png')
    plt.show()

#ab = np.ones((4, 4))
#ab_phase = np.arange(N*M).reshape(N,M)  * (30*np.pi/180)
#ab_phase = np.array([[0, 10, 20, 30],[0, 10, 20, 30],[0, 10, 20, 30]]) * (np.pi/180)
#ab_phase = (np.pi/2) + (np.random.randn(N,M) * np.pi/2)  # random phases: mean pi/2, from 0 until pi
#ab_phase = np.array([[0.0, 0.00013297, 0.00026595, 0.00039892], [0.00521822, 0.00535119, 0.00548416, 0.00561714], [0.01043643, 0.0105694, 0.01070238, 0.01083535], [0.01565465, 0.01578762, 0.01592059, 0.01    605357]])
#ab_coe = np.exp(1j * ab_phase)
patter3d()
