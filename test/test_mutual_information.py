import numpy as np
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt


def gen_channel(num_rx, num_tx, variance):
    sigma = variance
    h = np.sqrt(sigma/2)*(np.random.randn(num_rx, num_tx) + np.random.randn(num_rx, num_tx) * 1j)
    return h

if __name__ == "__main__":
    num_rx = 2
    num_tx = 2
    variance = 1.0
    num_samples = 1000
    norm_samples = []
    gain_channel = []
    gain_bf_opt = []
    gain_bf_est = []
    #gain_beamforming = []

    for _ in range(num_samples):
        h = gen_channel(num_rx, num_tx, variance)
        h = num_tx * h/norm(h)
        norm_samples.append(norm(h))

        u, s, vh = svd(h)
        gain_channel.append(s ** 2)


        f = np.matrix(vh.conj()[0,:]) # vetor linha: 1xN
        f_est = (1/np.sqrt(num_tx)) * np.exp(1j * np.angle(f))
        

        w = np.matrix(u.conj()[:,0]) # vetor coluna: Nx1

        print (f'\n\nAnalyzing procoder/combining: \n')

        print (f'np.shape(f): {np.shape(f)}')
        print (f'np.shape(f_est): {np.shape(f_est)}')
        print (f'norm(f): {norm(f)}')
        print (f'norm(f_est): {norm(f_est)}')
        print (f'f: \n{f}')
        print (f'f_est: \n{f_est}')
        print (f'np.angle(f) deg: {np.rad2deg(np.angle(f))}')
        print (f'np.angle(f_est) deg: {np.rad2deg(np.angle(f_est))}')
        print (f'np.abs(f): {np.abs(f)}')
        print (f'np.abs(f_est): {np.abs(f_est)}')

        print (f'\n')

        #print (f'np.shape(w): {np.shape(w)}')
        #print (f'norm(w): {norm(w)}')
        #print (f'w: \n{w}')
        #print (f'np.angle(w) deg: {np.rad2deg(np.angle(w))}')
        #print (f'np.abs(w): {np.abs(w)}')

        prod_opt = np.dot(w, np.dot(h, f.T))
        #print (np.shape(prod))
        gain_bf_opt.append(norm(prod_opt[0,0]) ** 2)

        prod_est = np.dot(w, np.dot(h, f_est.T))
        #print (np.shape(prod))
        gain_bf_est.append(norm(prod_est[0,0]) ** 2)
        #print(f'norm prod: {norm(prod)}')
        #print(f'norm h: {norm(h)}')

    gain_channel = np.array(gain_channel)
    gain_bf_opt = np.array(gain_bf_opt)
    gain_bf_est = np.array(gain_bf_est)
    #print(f'gain_s shape: {np.shape(gain_s)}')
    #plt.plot(norm_samples)
    plt.plot(np.sum(gain_channel, axis=1), label='overall channel')
    #plt.plot(np.sum(gain_s_prod, axis=1), label='prod overall')
    plt.plot(gain_channel[:,0], label='channel sigma0')
    #plt.plot(gain_s_prod[:,0], label='sigma0 prod')
    plt.plot(gain_bf_opt, label='opt bf gain')
    plt.plot(gain_bf_est, label='est bf gain')
    plt.plot(gain_bf_opt - gain_bf_est , label='diff (opt - est) bf gain')
    #plt.plot(gain_s[:,1], label='sigma1')
    plt.legend()
    plt.show()
