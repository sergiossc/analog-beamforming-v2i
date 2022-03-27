import sys
import numpy as np
from numpy.linalg import svd, norm
#from lib.vq.utils import norm
import matplotlib.pyplot as plt
from utils import richscatteringchnmtx


if __name__ == "__main__":
   
    norm_values = []

    num_rx = 4
    num_tx = 4
    variance = 1

    for i in range(100):
        pass
        ch = richscatteringchnmtx(num_rx, num_tx, variance)
        ch = np.sqrt(num_rx * num_tx) * (ch/norm(ch))
        norm_v = norm(ch)
        norm_values.append(norm_v)
    #   u, s, vh = svd(ch)
    #   s = s ** 2
    #   #print (s)

    plt.plot(norm_values)
    plt.show()
    #samples_filename_joined = f'{samples_pathfile[0:-4]}_joined.npy'
    #np.save(samples_filename_joined, np.array(channels_joined))
    #print (samples_filename_joined)
