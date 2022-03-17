import numpy as np
from numpy.linalg import norm

def get_rate(sample, cw, snr):
    #snr = 3.16 # 5dB
    #bw = 2 * (10 ** 9)
    gain = norm(np.matmul(sample, cw.conj().T))
    rate = np.log2(1 + snr * gain)
    return rate

