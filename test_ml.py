import numpy as np
from numpy.linalg import norm


def ml(y, u):
    est_a = norm(y - u['u_a'])
    est_b = norm(y - u['u_b'])
    if est_a < est_b:
        return u['u_a']
    else:
        return u['u_b']
# symbols
u_a = 1 * np.exp(1j * 0)
u_b = 1 * np.exp(1j * np.pi)

u = {'u_a': u_a, 'u_b': u_b}

# stream
lenght = 100
s = np.random.choice([u_a, u_b], lenght)

# noise
sigma = 1
w = np.sqrt(sigma/2) * np.random.rand(lenght) + np.sqrt(sigma/2) * 1j * np.random.rand(lenght)

# channel
h = np.sqrt(sigma/2) * np.random.rand(lenght) + np.sqrt(sigma/2) * 1j * np.random.rand(lenght)

# noiselly received signal
y_rcvd = s * h + w

# ml detection
y_est = []
for i in range(lenght):
    y_est.append(ml(y_rcvd[i], u))

y_est = np.array(y_est)


for i in range(lenght):
    print (f's_i: {s[i]} --> y_est: {y_est[i]}')
    if s[i] == y_est[i]:
        print (f'OK\n')
    else:
        print (f'WRONG\n')
