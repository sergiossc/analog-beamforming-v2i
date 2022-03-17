import numpy as np

n = 4 # num rx antennas
m = 4 # num tx antennas

cov_matrix = np.zeros((n,m))

theta_r = np.pi/10
d_r = 2 # in meters
s = 21

for i in range(n):
    for j in range(m):
        cov_matrix[i, j] = (1/s) * np.sum()
