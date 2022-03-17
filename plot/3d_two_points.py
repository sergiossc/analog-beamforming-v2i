import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


ax.scatter(0, 0, 0, marker='^')
ax.text(0, 0, 0, "Tx(0, 0, 0)", color='red')
ax.scatter(10, 0, 10, marker='^')
ax.text(10, 0, 10, "Rx(10, 0, 10)", color='red')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()