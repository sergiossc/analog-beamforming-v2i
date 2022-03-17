import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.constants import c
import sys
from utils import fftmatrix

def get_spatial_signature(nt, omega_t, d):
    e = (1/np.sqrt(nt)) * np.matrix([np.exp(-1j * 2 * np.pi * d * i * omega_t) for i in range(nt)])
    return e.T

if __name__ == '__main__':

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
    antenna_setup_set = [ (4,0.5), (8, 0.5), (16, 0.5), (2,1), (6, 0.3334), (32, 0.0625)]
    for antenna_set in antenna_setup_set:
        #nt = int(sys.argv[1])
        nt = antenna_set[0]
        theta_t = np.pi/2
        omega_t = np.cos(theta_t)
        #d = float(sys.argv[2])
        d = antenna_set[1]
        e_t = get_spatial_signature(nt, omega_t, d)
        #fftmat, ifftmat = fftmatrix(nt)
        #e_t = fftmat[:,0]
        print (e_t)
        print (np.shape(e_t))
        print (f'norm of e_t: {norm(e_t)}')
    
        theta = np.arange(0, 2*np.pi, 0.01)
        omega = np.cos(theta)
        psi = []
        
        for o in omega:
            e_x = get_spatial_signature(nt, o, d)
            v  = np.abs(e_t.conj().T * e_x)[0,0]
            psi.append(v)
    
        
        ax.plot(theta, psi, label=f'N={nt}, $\Delta=${d}, L={int(nt * d)}')
    plt.legend()
    #plt.show()
    fig_filename = f'ula-pattern.png'
    print (fig_filename)
    plt.savefig(fig_filename, bbox_inches='tight')

