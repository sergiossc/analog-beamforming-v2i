import numpy as np
from scipy.constants import c


fc = 60 * (10 ** 9) # freq in Hz
wavelength = c/fc # lambda


print (f'wavelength: {wavelength}')


N = 4 # number of elements

theta_d = np.pi/2
#w = np.array([np.e ** (1j * n * np.pi * np.cos(theta_d)) for n in np.arange(N)])
w = np.array([np.e ** (1j * n * theta_d) for n in np.arange(N)])

print (w)
print (np.rad2deg(np.angle(w)))

