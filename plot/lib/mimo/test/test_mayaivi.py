# Create the data.
import numpy as np
from scipy.constants import speed_of_light

def dft_codebook(dim):


dtheta = np.linspace(0, 2*np.pi, 100)
dphi = np.linspace(0, 2*np.pi, 100)
theta, phi = np.meshgrid(dtheta, dphi)

X = np.sin(theta)*np.cos(phi)
Y = np.sin(theta)*np.sin(phi)
Z = np.cos(theta)

tx_array_x = 4
tx_array_y = 4
tx_array_z = 1

fc = 60 * (10 ** 9)
wavelength = speed_of_light/fc
d = wavelength/2
k = 2 * np.pi * (1/wavelength)

af = 0 
for x in range(tx_array_x):
    for y in range(tx_array_y):
        for z in range(tx_array_z):
            print ('[x,y,z]: ', x, y, z)
            delay = (x * X) + (y * Y) + (z * Z)
            af = af + np.exp(1j * k * d * delay) 

af = np.abs(af)
X = af * X
Y = af * Y
Z = af * Z


# View it.
from mayavi import mlab
s = mlab.mesh(X, Y, Z)
mlab.show()
