from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.io import loadmat

#import matplotlib.pyplot as #plt
#import numpy as np

##f = loadmat('eigen_beam_direction_ch1.mat')
##eigen_beam_direction = f['eigen_beam_direction']
##print("eigen_beam_direction[0]: ", eigen_beam_direction[0])
##phi = eigen_beam_direction[:,3]
##print ("len(my_phi): ", len(phi))

phi = np.arange(100)

fig = plt.figure()

ax = fig.gca(projection='3d')
nphi=48
nth=12

#phi = np.linspace(0,360, nphi)/180.0*np.pi
th = np.linspace(-90,90, nth)/180.0*np.pi

verts2 = []
for i  in range(len(phi)-1):
    for j in range(len(th)-1):
        r= np.cos(phi[i]**2)     #  <----- your function is here
        r1= np.cos(phi[i+1])**2
        cp0= r*np.cos(phi[i])
        cp1= r1*np.cos(phi[i+1])
        sp0= r*np.sin(phi[i])
        sp1= r1*np.sin(phi[i+1])

        ct0= np.cos(th[j])
        ct1= np.cos(th[j+1])

        st0=  np.sin(th[j])
        st1=  np.sin(th[j+1])

        verts=[]
        verts.append((cp0*ct0, sp0*ct0, st0))
        verts.append((cp1*ct0, sp1*ct0, st0))
        verts.append((cp1*ct1, sp1*ct1, st1))
        verts.append((cp0*ct1, sp0*ct1, st1))
        verts2.append(verts   )

poly3= Poly3DCollection(verts2, facecolor='g')  

poly3.set_alpha(0.2)
ax.add_collection3d(poly3)
ax.set_xlabel('X')
ax.set_xlim3d(-1, 1)
ax.set_ylabel('Y')
ax.set_ylim3d(-1, 1)
ax.set_zlabel('Z')
ax.set_zlim3d(-1, 1)


plt.show()
#from mpl_toolkits.mplot3d import Axes3D
#from scipy.io import loadmat
#import matplotlib.pyplot as plt
#import numpy as np

#f = loadmat('eigen_beam_direction_ch1.mat')
#eigen_beam_direction = f['eigen_beam_direction']

#beam = np.meshgrid(eigen_beam_direction)
