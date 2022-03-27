from utils import richscatteringchnmtx
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

if __name__ == "__main__":

    n_list = [4, 8, 16, 32, 64]


    num_samples = 1000
    variance = 1.0

    for n in n_list:

        num_rx = n
        num_tx = n
        samples = []
        norm_h = []
        for _ in range(num_samples):
            h = richscatteringchnmtx(num_rx, num_tx, variance)
            h = np.sqrt(num_rx * num_tx) * h/norm(h)
            #norm_h.append(norm(h))
            #print (f'norm: {norm(h)}')
            samples.append(h)
        samples = np.array(samples)
        #plt.plot(norm_h)
        #plt.show()
        print (n)
        np.save(f'samples_random_{num_rx}x{num_tx}.npy', samples)
