'''
https://www.mathworks.com/matlabcentral/fileexchange/27932-3d-array-factor-of-a-4x4-planar-array-antenna
'''
import sys
sys.path.append(r'/home/snow/github/land/lib/mimo_tools/')
sys.path.append(r'/home/snow/github/land/lib/mimo_tools/utils')
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from scipy.constants import c
from mpl_toolkits import mplot3d
from numpy.linalg import norm
from phased import PartitionedArray
#class PartitionedArray:
#    def __init__(self, size, d, num_rf, wavelength, type_of="ULA"):
#        self.size = size
#        self.d = d
#        self.num_rf = num_rf
#        self.wavelength = wavelength
#        self.type_of = type_of
#        self.ula = np.arange((size))

def patter3d(phased_array, ab_coe):

    #fc = 60 * (10 ** 9)
    wavelength = phased_array.wave_length
    d = phased_array.element_spacing
    k = 2 * np.pi * 1/wavelength

    length = 200  # of samples
    theta0 = np.linspace(0, 2*np.pi, length) # Elevation angles
    phi0 = np.linspace(0, 2*np.pi, length) # Azimuth angles
    
    #theta, phi = np.meshgrid(theta0, phi0)
    phi, theta = np.meshgrid(phi0, theta0)
    
    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)
    
    N = int(np.sqrt(phased_array.size))
    M = int(np.sqrt(phased_array.size))
    
   
    af = 0
    for n in range(N):
        for m in range(M):
            af = ab_coe[n][m] * np.exp(-1j * k * d * n * u) * np.exp(-1j * k * d * m * v) + af
            #af = np.exp(-1j * k * d * n * u) * np.exp(-1j * k * d * m * v) + af
            
    
    
    total_power = np.sum(ab ** 2)
    af_power = af ** 2
    normalized_af = np.abs(af_power/total_power) * 1/phased_array.size
    
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



fc = 60 * (10 ** 9)
_wavelength = c/fc
_d = _wavelength/2
 
# Uniform Planar Array Setup
num_tx = 64
num_tx_rf = 6
tx_array = PartitionedArray(num_tx, _d, num_tx_rf, _wavelength, "UPA") # configuracao da antena de transmissao
#(self, size, element_spacing, num_rf, wave_length, formfactory):

# w
N = int(np.sqrt(tx_array.size))
M = int(np.sqrt(tx_array.size))
ab = np.ones((N, M))
np.random.seed(444)
ab_phase = (np.pi/2) + (np.random.randn(N,M) * np.pi/2)  # random phases: mean pi/2, from 0 until pi
ab_coe = ab * np.exp(1j * ab_phase)
 
 
patter3d(tx_array, ab_coe)
