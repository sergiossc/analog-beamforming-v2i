import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from scipy.io import loadmat

#import matplotlib.pyplot as #plt
#import numpy as np


f = loadmat('eigen_beam_direction_ch1.mat')
eigen_beam_direction = f['eigen_beam_direction']
print("eigen_beam_direction[0]: ", eigen_beam_direction[0])
phi = eigen_beam_direction[:,3]
print ("len(my_phi): ", len(phi))

F = eigen_beam_direction[:,0]
print ('F.shape:\n', F.shape)
print ('F:\n', F)
theta, phi = np.linspace(0, 2 * np.pi, 64), np.linspace(0, np.pi, 64)
THETA, PHI = np.meshgrid(theta, phi)

X = np.sin(PHI) * np.cos(THETA)
Y = np.sin(PHI) * np.sin(THETA)

N = int(np.sqrt(len(F)))
M = int(np.sqrt(len(F)))
w = np.array(F).reshape((N,M))

af = 0
for n in range(N):
    for m in range(M):
        af = w[n][m] * np.exp(-1j * n * X) * np.exp(-1j * m * Y) + af
        #af = np.exp(-1j * k * d * n * u) * np.exp(-1j * k * d * m * v) + af


af = np.abs(af)

X = af * np.sin(PHI) * np.cos(THETA)
Y = af * np.sin(PHI) * np.sin(THETA)
Z =  af * np.cos(PHI)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
plot = ax.plot_surface(X, Y, af, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),linewidth=0, antialiased=False, alpha=0.5)
plt.show()
