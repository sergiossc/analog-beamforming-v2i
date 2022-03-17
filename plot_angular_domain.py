import sys
import argparse
import numpy as np
from numpy.linalg import norm, svd
from scipy.constants import c
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import array_to_angular_domain
#from utils import get_spatial_signature2, get_spatial_signature, get_orthonormal_basis, plot_angular_domain, gen_channel
#import seaborn as sns; sns.set_theme()



if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_group = my_parser.add_mutually_exclusive_group(required=True)
    my_group.add_argument('-may', '--mayavi', action='store_true', help=f'if you chose plot option \'mayavi\'' )
    my_group.add_argument('-mat', '--matplotlib', action='store_true', help=f'if you chose plot option \'matplotlib\'' )
    args = my_parser.parse_args()
   

    #samples_file = f'/home/snow/github/land/dataset/npy_files_s007/training_set_64x64.npy'
    samples_file = f'S000/s000-training_set_64x64_a.npy'
    samples = np.load(samples_file)
    num_samples, num_rx, num_tx = np.shape(samples)
    nr = num_rx
    nt = num_tx
    sample_index = np.random.choice(num_samples, replace=True)
    #sample_index = 958
    print (f'sample_index: {sample_index}')
    #h = samples[sample_index]
    h = samples[sample_index]
    #h = gen_channel(nr, nt, 1.0)
    ##h = nt * h/norm(h)
    #Getting orthonormal basis
    ##fc = 60 * (10 ** 9)
    ##wavelength = c/fc
    ##d = 1/2
    ##Lt = nt * d
    ##Lr = nr * d


    ##s_rx = get_orthonormal_basis(nr, d, wavelength)
    ##s_tx = get_orthonormal_basis(nt, d, wavelength)
    ##h_a = s_rx.conj().T * (h * s_tx.conj())
    #h_a = array_to_angular_domain(h)
    h_a = h
    x = np.arange(nr)
    y = np.arange(nt)
    X, Y = np.meshgrid(x, y)
    Z = np.abs(h)
    # View it.
    if args.mayavi == True and args.matplotlib == False:
        from mayavi import mlab
        s = mlab.mesh(X, Y, Z)
        #s = mlab.barchart(Z)
        #mlab.axes(x_axis_visibility=True)
        mlab.show()
        
    elif args.mayavi == False and args.matplotlib == True:
        fig = plt.figure(figsize=(10, 6))
    
        ax1 = fig.add_subplot(111, projection='3d')
        #surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax1.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
        ax1.set_xlabel('RX bins')
        ax1.set_ylabel('TX bins')
    
        #ax1.set_title('gist_earth color map')
        surf1 = ax1.plot_surface(X, Y, Z, cmap=cm.twilight_shifted)
    
        #fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        ax1.set_zlabel(r'|H$^{a}$|', fontweight="bold")
    
        plt.show()
    else:
        print ('Use -h to see options')
        sys.exit()
