import sys
sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/mimo')
sys.path.append(r'/home/snow/analog-beamforming-v2i/lib/utils')
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from scipy.constants import c
from phased import PartitionedArray
from plot import pattern3d


def dft_codebook(dim):
    seq = np.matrix(np.arange(dim))
    mat = seq.conj().T * seq
    w = np.exp(1j * 2 * np.pi * mat / dim)
    return w


def calc_omega(theta, phi, k, d):
    omegax = k * d * np.sin(theta) * np.cos(phi)
    omegay = k * d * np.sin(theta) * np.sin(phi)
    omegaz = k * d * np.cos(theta)
    return omegax, omegay, omegaz

class path:
    def __init__(self, id, aod_theta, aod_phi, aoa_theta, aoa_phi, phase, received_power):
        self.id = str(id)
        self.aod = {'theta':np.deg2rad(aod_theta), 'phi': np.deg2rad(aod_phi)}
        self.aoa = {'theta': np.deg2rad(aoa_theta), 'phi': np.deg2rad(aoa_phi)}
        self.received_power = np.power(10,received_power/10) # convert dB in W
        self.phase = phase
        self.at = []
       
    def to_string(self):
        print ('id: ', self.id)
        print ('aod[theta, phi]: ', np.rad2deg([self.aod['theta'], self.aod['phi']]))
        print ('aoa[theta, phi]: ', np.rad2deg([self.aoa['theta'], self.aoa['phi']]))
        print ('received_power: ', self.received_power)
        print ('path phase: ', self.phase)


fc = 60 * (10 ** 9)
wavelength = c/fc
d = wavelength/4
k = 2 * np.pi * (1/wavelength)

# Uniform Planar Array 
num_tx = 25
num_tx_rf = 1
tx_array = PartitionedArray(num_tx, d, num_tx_rf, wavelength, "UPA") # configuracao da antena de transmissao 
num_rx = 4
num_rx_rf = 1
rx_array = PartitionedArray(num_rx, d, num_rx_rf, wavelength, "UPA") # configuracao da antena de recepcao 

tx_ura = tx_array.ura
rx_ura = rx_array.ura
print ('tx_ura:\n', tx_ura)
print ('rx_ura:\n', rx_ura)


# Setting up random paths
path_spacing = 100
theta0 = np.linspace(0, np.pi, path_spacing) # Elevation angles
phi0 = np.linspace(0, 2*np.pi, path_spacing) # Azimuth angles

num_paths = 1
paths = []
#np.random.seed(222)
for i in range(num_paths):
    #aod_theta = np.rad2deg(np.random.choice(theta0)-(np.pi/2))
    #aod_phi = np.rad2deg(np.random.choice(phi0))
   
    aod_theta = 90
    aod_phi = 0

    aoa_theta = np.random.choice(theta0)
    aoa_phi = np.random.choice(phi0)

    received_power = np.random.choice([0], 1, replace=False)  # same power for all paths: 1W
    phase = 2*np.pi

    p = path(i, aod_theta, aod_phi, aoa_theta, aoa_phi, phase, received_power)
    paths.append(p)

# at transmition process...
aod_theta = [p.aod['theta'] for p in paths]
aod_phi = [p.aod['phi'] for p in paths]

# at receiving process...
aoa_theta = [p.aoa['theta'] for p in paths]
aoa_phi = [p.aoa['phi'] for p in paths]


#phi, theta = np.meshgrid(aod_phi, aod_theta)
#print ('phi.shape: ', phi.shape)
#print ('theta.shape: ', theta.shape)

departure_omegax, departure_omegay, departure_omegaz = calc_omega(aod_theta, aod_phi, k, d)
##arrival_omegax, arrival_omegay, arrival_omegaz = calc_omega(aoa_theta, aoa_phi, k, d)
print('departure_omegax: ', departure_omegax)
print('departure_omegay: ', departure_omegay)
print('departure_omegaz: ', departure_omegaz)
 
# @ TX
num_tx_x = int(np.sqrt(1)) # number of elelement of tx at x_axis
num_tx_y = int(np.sqrt(tx_array.size)) # number of elelement of tx at y_axis
num_tx_z = int(np.sqrt(tx_array.size)) # number of elelement of tx at z_axis

upa_delay = np.zeros((num_tx_x, num_tx_y, num_tx_z), dtype=complex)
# Getting AF of Planar antenna..
for n in range(len(paths)):    
    for num_x in range(num_tx_x):
        for num_y in range(num_tx_y):
            for num_z in range(num_tx_z):
                print ('--------------------------')
                print('num_x: ', num_x)
                print('num_y: ', num_y)
                print('num_z: ', num_z)
                delay = (num_x * departure_omegax[n]) + (num_y * departure_omegay[n]) + (num_z * departure_omegaz[n])
                print('delay: ', delay)
                upa_delay[num_x, num_y, num_z] = np.exp(1j * delay)
    
print ('upa_delay.abs: \n', np.abs(upa_delay))
print ('upa_delay.phase: \n', np.rad2deg(np.angle(upa_delay)))
pattern3d(tx_array, upa_delay.conj().T, 'None')

### @ RX
##num_rx_x = int(np.sqrt(1)) # number of elelement of rx at x_axis
##num_rx_y = int(np.sqrt(rx_array.size)) # number of elelement of rx at y_axis
##num_rx_z = int(np.sqrt(rx_array.size)) # number of elelement of rx at y_axis
##
### Stimating the channel...
##complex_gain = [p.received_power*np.exp(1j * p.phase) for p in paths]
##received_power_list = [p.received_power for p in paths]
##factor = np.sqrt(num_rx * num_tx) * (1/len(paths)) # * (np.linalg.norm(complex_gain)/np.sum(complex_gain))
##
##ab = np.zeros((num_tx_x, num_tx_y, num_tx_z), dtype=complex)
##for n in range(len(paths)):
##    paths[n].to_string()
##    ab_est = np.zeros((num_tx_x, num_tx_y, num_tx_z), dtype=complex)
##    for x in range(num_tx_x):
##        for y in range(num_tx_y):
##            for z in range(num_tx_z):
##            
##                #print ('--------------------------')
##                #print('x: ', x)
##                #print('y: ', y)
##                #print('z: ', z)
##                #print('departure_omegax[n]: ', departure_omegax[n])
##                #print('departure_omegay[n]: ', departure_omegay[n])
##                delay = (y * departure_omegax[n]) + (z * departure_omegay[n]) + (x * departure_omegaz[n])
##                #print('delay: ', delay)
##                ##ab_est[x, y, z] = delay
##                ab_est[x, y, z] = 1 * np.exp(1j * delay)
##        ab = ab + ab_est
##
##
###h = factor * h
##print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
##ab = ab[0]
##print ('ab.shape:\n', ab.shape )
##print ('ab:\n', ab )
##print ('ab.abs:i\n', np.abs(ab))
##print ('ab.angle:\n', np.rad2deg(np.angle(ab)))
###opt_w = np.array(unitary_w).reshape(int(np.sqrt(tx_array.size)), int(np.sqrt(tx_array.size)))
##
###pattern3d(tx_array, ab, 'title')
##
##
##h = np.zeros((rx_array.size, tx_array.size), dtype=complex)
##for n in range(len(paths)):
##    paths[n].to_string()
##    # @ TX
##    tx_vecx = np.exp(1j * departure_omegay[n] * np.arange(num_tx_z))
##    tx_vecy = np.exp(1j * departure_omegax[n] * np.arange(num_tx_y))
##    at = 1/np.sqrt(num_tx_z * num_tx_y) * np.kron(tx_vecy, tx_vecx)
##    at = np.array(at).reshape(num_tx,1)
##    print ('at.shape: ', at.shape)
##
##    # @ RX
##    rx_vecx = np.exp(1j * arrival_omegay[n] * np.arange(num_rx_z))
##    rx_vecy = np.exp(1j * arrival_omegax[n] * np.arange(num_rx_y))
##    ar = 1/np.sqrt(num_rx_x * num_rx_y) * np.kron(rx_vecy, rx_vecx)
##    ar = np.array(ar).reshape(num_rx,1)
##    print ('ar.shape: ', ar.shape)
##    
##    # Channel contrib of this path:
##    h_contrib = np.matmul(ar, at.conj().T)
##    print('h_contrib.shape:\n', h_contrib.shape)
##
##    #h = h + complex_gain[n] * h_contrib
##    h = h + h_contrib
##
##h = factor * h
##
##h = np.array(h).reshape(num_tx_z, num_tx_y)
###pattern3d(tx_array, h.T)
##
##dim = num_tx
##dft_w = dft_codebook(dim)
##for n in range(dim):
##    w = dft_w[n]
##    w = np.array(w).reshape(num_tx_z, num_tx_y)
##    pattern3d(tx_array, w, 'dft')
