import sys
import argparse
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
from matplotlib import cm

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_group = my_parser.add_mutually_exclusive_group(required=True)
    my_group.add_argument('-may', '--mayavi', action='store_true', help=f'if you chose plot option \'mayavi\'' )
    my_group.add_argument('-mat', '--matplotlib', action='store_true', help=f'if you chose plot option \'matplotlib\'' )
    args = my_parser.parse_args()

    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    
    t, p = np.meshgrid(theta, phi)
    
    X = np.sin(t) * np.cos(p)
    Y = np.sin(t) * np.sin(p)
    Z = np.cos(t)
    
    fc = 60 * (10 ** 9)
    wavelength = c/fc
    
    k = 2 * np.pi * (1/wavelength)
    d = wavelength/2
    #psi = k * d * np.cos(theta)
    
    #plt.polar(theta, np.abs(psi))
    #plt.show()
    
    tx_array_x = 1
    tx_array_y = 1
    tx_array_z = 10
    
    af = 0
    #beta = -k * d
    #beta = np.deg2rad(45)
    beta = 0
    
    for x in np.arange(tx_array_x):
        for y in np.arange(tx_array_y):
            for z in np.arange(tx_array_z):
                delay = (x * X) + (y * Y) + (z * Z) + (x * np.deg2rad(30)) + (y * np.deg2rad(30)) + (z * beta)
                af = af + np.exp(1j * k * d * delay)
    af = np.abs(af)
    X = af * X
    Y = af * Y
    Z = af * Z




    # View it.

    if args.mayavi == True and args.matplotlib == False:
        from mayavi import mlab
        s = mlab.mesh(X, Y, Z)
        mlab.axes(x_axis_visibility=True)
        mlab.show()
        
    elif args.mayavi == False and args.matplotlib == True:
        fig = plt.figure(figsize=(10, 6))
    
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
    
    #mycmap = plt.get_cmap('gist_earth')
    
        ax1.set_title('gist_earth color map')
        surf1 = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
        plt.show()
    else:
        print ('Use -h to see options')
        sys.exit()
