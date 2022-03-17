from scipy.io import loadmat
#import matplotlib.pyplot as #plt
#import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
for i in range(4):

    f = loadmat('eigen_beam_direction_ch1.mat')
    eigen_beam_direction = f['eigen_beam_direction']
    print("eigen_beam_direction[0]: ", eigen_beam_direction[0])
    my_phi = eigen_beam_direction[:,i]
    print ("len(my_phi): ", len(my_phi))
    
    #theta, phi = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 40)
    theta, phi = np.linspace(0, 2 * np.pi, 64), np.linspace(0, np.pi, 64)
    #print("theta: ", theta)
    #print("phi: ", phi)
    THETA, PHI = np.meshgrid(theta, phi)
    #print("THETA: ", THETA)
    #print("PHI: ", PHI)
    R = np.cos(np.abs(my_phi))
    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    plot = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
        linewidth=0, antialiased=False, alpha=0.5)
    
plt.show()
