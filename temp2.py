import sys
import math
import load_lib
import numpy as np
import matplotlib.pyplot as plt
import lib.mimo.arrayconfig as phased_array

def PParttern(tx_array, w):
    #class path:
    #    def __init__(self, id, aod):
    #        self.id = id
    #        self.aod = np.deg2rad(aod)
    #        self.at = None
    
    theta = np.arange(0,360)
    
    num_tx_elements = tx_array.size
    wavelength = tx_array.wave_length  #in meters
    element_spacing = tx_array.element_spacing #in meters
    array_vec = np.arange(num_tx_elements)
    k = 2 * np.pi * 1/wavelength
    #at_res = np.array([np.exp(-1j * 2 * np.pi * element_spacing * (1/wavelength) * np.cos(np.angle(aod))) for aod in aods])
    #at_res = np.array([np.exp(-1j * np.angle(aod)) for aod in aods])
    at_res = w
    
    prod = np.zeros(len(theta), dtype=complex)
    f_psi = np.zeros(len(theta), dtype=complex)
    psi = np.zeros(len(theta))
    for i in range(len(theta)):

        t = np.deg2rad(theta[i])
        p = k * element_spacing * np.cos(t)
        psi[i] = p
        af = np.array([np.exp(1j * n * p) for n in range(len(array_vec))])

        af0 = np.sum(af) 
        af1 = af0 * np.exp(1j * p)
        af2 = af0 - af1 
        af3 = af2/(1 - np.exp(1j * p))
        f_psi[i] = af3 #/len(array_vec)

        #product = np.matmul(at_res.T, af)
        #product = product/np.matmul(at_res.conj().T, at_res)#print ('product: \n', product)
        #prod[i] = product #/len(array_vec)
    #plt.plot(theta, prod)
    #plt.legend()
    #plt.savefigfig('steering.png')
    #plt.show()
    fig = plt.figure()
    #p = np.random.rand(100) * 1j + np.random.rand(100)
    #angles = np.angle(f_psi)
    #mag = np.abs(f_psi)
    fig.add_subplot(111, projection='polar')
    #fig.add_subplot(111)
    plt.plot(psi, np.abs(f_psi))
    #plt.polar(psi, f_psi)
    #plt.title(str(id_scene))
    plt.show()

def plot2():
    #from pylab import *
    #%matplotlib inline
    #Varable Declaration
    
    N=5   #Number of elements of dipole
    s=0.25 #Space between dipole elements(wavelengths)
    phi0=90*math.pi/180 #Angle between array factor and array(radians)
    
    #Calculation
    
    alpha=-2*math.pi*s*math.cos(phi0)  #Current phase(radians)
    #phi=np.arange(-180,185,5)
    phi = np.linspace(0, 360, 73)
    print (len(phi))
    Si=np.linspace(0,0,73)
    for k in range(0,73):
        Si[k]=alpha+2*math.pi*s*math.cos(phi[k]*math.pi/180)
    
    AFR=np.linspace(0,0,73)
    AFI=np.linspace(0,0,73)
    
    for i in range(0,73):
      for j in range(0,N):
         AFR[i]=AFR[i]+math.cos(j*Si[i])  #Real part of Array factor
         AFI[i]=AFI[i]+math.sin(j*Si[i])#Imaginary part of Array factor
    
    teta=phi*math.pi/180
    AF=np.linspace(0,0,73)
    for k in range(0,73):
       AF[k]=AF[k]+(AFR[k]**2+AFI[k]**2)**0.5
    
    #Result
    
    plt.polar(teta,AF)
    #title('Polar plot of Array Factor')
    plt.show()
    
#tx_array = phased_array.tx_array
#PParttern(tx_array, None)
#plot2()
from scipy.constants import c
fc = 60 * (10 ** 9)
array_size = 4
wavelength = c/fc
d = wavelength/2
theta = np.linspace(-np.pi, np.pi, 300) #anlg
array_vec = np.arange(array_size)

psi = np.zeros(len(theta))
f_psi = np.zeros(len(theta), dtype=complex)
af = np.zeros(len(theta), dtype=complex)
b = 0 # beta
for i in range(len(theta)):
    t = theta[i]
    psi[i] = 2 * np.pi * (1/wavelength) * d * np.cos(t + b)
    f_psi[i] = np.sum(np.array([np.exp(1j * n * psi) for n in array_vec]))
    #af[i] = f_psi[i]  * (np.exp(1j * psi[i]))
    af[i] = (1 - np.exp(1j * array_size * psi[i])) / (1 - np.exp(1j * psi[i]))
    #af[i] = ((f_psi[i] * np.exp(1j * psi[i])) - f_psi[i]) / (1 - np.exp(1j * array_size * psi[i]))# 

#psi = np.linspace(-np.pi, np.pi, 100) #anlg
#f_psi = np.cos(psi) #gain
fig = plt.figure()
fig.add_subplot(111, projection='polar')
plt.plot(psi, np.abs(af)/array_size)
plt.show()
