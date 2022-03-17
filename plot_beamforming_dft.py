import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.constants import c
import sys
from utils import fftmatrix, get_spatial_signature

#def get_spatial_signature(nt, omega_t, d):
#    e = (1/np.sqrt(nt)) * np.matrix([np.exp(-1j * 2 * np.pi * d * i * omega_t) for i in range(nt)])
#    return e.T

if __name__ == '__main__':
    nt = int(sys.argv[1])
    q = int(sys.argv[2]) # oversampling factor
    #nt = 64
    theta_t = np.pi/2
    omega_t = np.cos(theta_t)
    #d = float(sys.argv[2])
    d = 0.5
    #e_t = get_spatial_signature(nt, omega_t, d)
    fftmat, ifftmat = fftmatrix(nt, q)
    #e_t = fftmat[:,3]
    #print (e_t)
    #print (np.shape(e_t))
    #print (f'norm of e_t: {norm(e_t)}')

    theta = np.arange(0, 2*np.pi, 0.01)
    omega = np.cos(theta)
    psi_mat = np.zeros((q*nt, len(omega)))
    for n in range(q*nt):
        e_t = fftmat[n].T
        for o in range(len(omega)):
            e_x = get_spatial_signature(nt, omega[o], d)
            v  = np.abs(e_t.conj().T * e_x)[0,0]
            psi_mat[n,o] = v

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    for n in range(q*nt):
        ax.plot(theta, psi_mat[n,:])
    fig_filename = f'dft_codebook_n{nt}_q{q}'
    plt.savefig(fig_filename, bbox_inches='tight')
    print (fig_filename)
    #plt.show()
