from utils import richscatteringchnmtx
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_rx = 128
    num_tx = 128
    num_samples = 100
    samples = []
    norm_h = []
    for _ in range(num_samples):
        variance = 1.0
        h = richscatteringchnmtx(num_rx, num_tx, variance)
        h = num_tx * h/norm(h)
        norm_h.append(norm(h))
        #print (f'norm: {norm(h)}')
        samples.append(h)
    samples = np.array(samples)
    #plt.plot(norm_h)
    #plt.show()
    np.save(f'samples_random_{num_rx}x{num_tx}.npy', samples)
