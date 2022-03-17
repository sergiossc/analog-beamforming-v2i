"""
    cite: https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
"""
from mpl_toolkits import mplot3d
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm


def f(x, y):
    #return np.sin(np.sqrt(x ** 2 + y ** 2))
    #return np.sin(np.sqrt(x ** 2 + y ** 2))
    return x + y 

#x = np.linspace(-6, 6, 30)
theta = np.linspace(0, np.pi/2, 100)
#y = np.linspace(-6, 6, 30)
phi = np.linspace(0, 2*np.pi, 100)
#y =  x  #np.linspace(-6, 6, 30)

u = np.sin(theta) + np.cos(phi)
v = np.sin(theta) + np.sin(phi)

U, V = np.meshgrid(u, v)

print ('U.shape:\n', U.shape)
print ('U:\n', U)
print ('V.shape:\n', V.shape)
print ('V:\n', V)

Z = f(U, V)

print ('Z.shape:\n', Z.shape)
print ('Z:\n', Z)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(U, V, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title('surface');

#ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('U')
ax.set_ylabel('V')
ax.set_zlabel('Z')

plt.show()
