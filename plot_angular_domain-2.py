import sys
import argparse
import numpy as np
from numpy.linalg import norm, svd
from scipy.constants import c
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns; sns.set_theme()

def get_spatial_signature2(n, d, wavelength, omega):
    spatial_signature = (1/np.sqrt(n)) * np.matrix([np.exp(1j * 2 * np.pi * d/wavelength * i * omega) for i in range(n)])
    return spatial_signature

def get_spatial_signature(n, d, wavelength, omega):
    spatial_signature = (1/np.sqrt(n)) * np.matrix([np.exp(-1j * 2 * np.pi * d/wavelength * i * omega) for i in range(n)])
    return spatial_signature

def get_orthonormal_basis(n, d, wavelength):
    L = n * d
    s_set = np.zeros((n,n), dtype=complex)
    #slots = np.arange(-n/2, n/2 + 1)
    slots = np.arange(n)
    for i in range(n):
        e = get_spatial_signature2(n, d, wavelength, slots[i]/L)
        s_set[i:] = e
    s_set = np.matrix(s_set)
    return s_set

def plot_angular_domain(nr, Lr, nt, Lt, paths):
    s_rx = np.zeros(nr)
    s_tx = np.zeros(nt)
    for p in paths:
        pathgain = p['pathgain']
        aoa = p['aoa']
        aod = p['aod']
        k_index = None
        l_index = None
        for k in range(nr):
            if np.cos(aoa) > (k/Lr - 1/(2*Lr)) and np.cos(aoa) <= (k/Lr + 1/(2*Lr)):
                k_index = k
        for l in range(nt):
            if np.cos(aod) > (l/Lt - 1/(2*Lt)) and np.cos(aod) <= (l/Lt + 1/(2*Lt)):
                l_index = l
    print (f'k_index = {k_index}')
    print (f'l_index = {l_index}')

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_group = my_parser.add_mutually_exclusive_group(required=True)
    my_group.add_argument('-may', '--mayavi', action='store_true', help=f'if you chose plot option \'mayavi\'' )
    my_group.add_argument('-mat', '--matplotlib', action='store_true', help=f'if you chose plot option \'matplotlib\'' )
    args = my_parser.parse_args()
   
    nt = 16
    nr = 16
    fc = 60 * (10 ** 9)
    wavelength = c/fc
    d = 1/2
    Lt = nt * d
    Lr = nr * d

    #Getting paths
    paths = []
    num_paths = 1
    for _ in range(num_paths):
        pathgain = np.random.rand() * np.exp(1j * 0) 
        aoa = np.random.choice(np.linspace(-np.pi, np.pi, num=100))
        aod = np.random.choice(np.linspace(-np.pi, np.pi, num=100))
        #aod = np.random.choice(np.linspace(-np.pi/2, np.pi/2, num=100))
        print (f'-> cos.oad: {np.cos(aod)} , cos.aoa: {np.cos(aoa)}')
        path = {'pathgain': pathgain, 'aoa': aoa, 'aod': aod}
        paths.append(path)

    h = np.zeros((nr, nt), dtype=complex)
    for p in paths:
        pathgain = p['pathgain']
        aoa = p['aoa']
        e_r = get_spatial_signature(nr, d, wavelength, np.cos(aoa))
        aod = p['aod']
        e_t = get_spatial_signature(nt, d, wavelength, np.cos(aod))
        h += pathgain * (e_r.T * e_t.conj())
    h = np.sqrt(nr * nt) * h

    h_a = np.zeros((nr, nt), dtype=complex)

    print (np.shape(h))
    #u, s, v = svd(h)
    #print (s)

    #Getting orthonormal basis
    s_rx = get_orthonormal_basis(nr, d, wavelength)
    s_tx = get_orthonormal_basis(nt, d, wavelength)
##
###    k = np.linspace(-1, 1, num=nr)
###    l = np.linspace(-1, 1, num=nt)
##    #h_a = np.zeros((nr, nt), dtype=complex)
##    #for ki in range(nr):
##    #    for li in range(nt):
##    #        e_rx = get_spatial_signature2(nr, d, wavelength, k[ki]/Lr)
##    #        e_tx = get_spatial_signature2(nt, d, wavelength, l[li]/Lt)
##    #        h_a[ki, li] = e_rx.conj() * (h * e_tx.conj().T) 
##
    h_a = s_rx.conj().T * (h * s_tx.conj())
    #z = h_a
    #ax = sns.heatmap(np.abs(z))
    #ax.plot()
    #plt.show()    
    #x = np.arange(nr)
    x = np.linspace(-1, 1, num=nr)
    #y = np.arange(nt)
    y = np.linspace(-1, 1, num=nt)
    X, Y = np.meshgrid(x, y)
    #u, s, v = svd(h_a)
    #print (u.shape)
    #print (np.rad2deg(np.angle(u)))
    #print (np.abs(u))
    #X = u
    #Y = v
    #Z = np.abs(u + v)
    Z = np.abs(h_a)




    # View it.

    if args.mayavi == True and args.matplotlib == False:
        from mayavi import mlab
        #s = mlab.mesh(X, Y, Z)
        s = mlab.barchart(Z)
        #mlab.axes(x_axis_visibility=True)
        mlab.show()
        
    elif args.mayavi == False and args.matplotlib == True:
        fig = plt.figure(figsize=(10, 6))
    
        ax1 = fig.add_subplot(111, projection='3d')
        #surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax1.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
    
        ax1.set_title('gist_earth color map')
        surf1 = ax1.plot_surface(X, Y, Z, cmap=cm.viridis)
    
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
        plt.show()
    else:
        print ('Use -h to see options')
        sys.exit()
