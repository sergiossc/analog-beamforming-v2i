import numpy as np
from numpy.linalg import svd
from lib.vq.utils import norm
import matplotlib.pyplot as plt

#def complex_average(samples):
#    return np.mean(samples, axis=0)
#
#def plot_samples(samples, filename, title, y_label):
#    fig, ax = plt.subplots()
#    ax.scatter(x=np.arange(len(samples)), y=np.abs(samples), marker='o', c='r', edgecolor='b')
#    ax.set_xlabel('samples')
#    ax.set_ylabel(y_label)
#    plt.title(title)
#    plt.savefig(filename)
#
#def plot_polar_samples(samples, filename):
#    nsamples, nrows, ncols = samples.shape
#    fig, axes = plt.subplots(nrows, ncols, subplot_kw=dict(polar=True))
#
#    for col in range(ncols):
#        axes[col].plot(0, 1, 'wo')
# 
#    for n in range(nsamples):
#        for col in range(ncols):
#            a = np.angle(samples[n, 0, col])
#            r = np.abs(samples[n, 0, col])
#            axes[col].plot(a, r, 'o')
#    plt.savefig(filename)


def waterfilling(s, snr):
    r = len(s)
    p = 1
    p_opt = np.zeros(r)
    while True:
        my_sum = 0
        for i in range(r-p+1):
            my_sum = my_sum + 1/s[i]
        mu = (1/(r-p+1)) * ( 1 + (1/snr) * my_sum )
        #p_opt = np.zeros(r-p+1)
        for i in range(r-p+1):
            p_opt[i] = mu - (1/(snr * s[i])) 
            if p_opt[i] < 0:
                p_opt[i] = 0
        print (p_opt)
        p = p + 1
        if p > 10:
            break
    return p_opt        


if __name__ == "__main__":
   
    
    # getting samples of channel
    print("Getting some samples of channel...")
    
    channels = np.load('/home/snow/github/land/dataset/npy_files_s007/training_set_4x4.npy')
    ch_id = np.random.choice(len(channels)) 
    my_ch = channels[ch_id]
    my_ch = 4 * my_ch
    #print ('channels.shape: ', channels.shape)
    #print ('ch.shape: ', ch.shape)

    snr_db = np.arange(-20, 20, 0.01)
    snr = 10 ** (snr_db/10)
    
    c_single_mode = np.zeros(len(snr))
    c_equal_power = np.zeros(len(snr))
    for ch in channels:
        ch = 4 * ch
        u, s, vh = svd(ch)
        s = s ** 2
        #print (s)
        c_single_mode = c_single_mode + np.log2(1 + snr * s[0])
    
        c_equal_power = c_equal_power + np.sum([np.log2(1 + snr * si) for si in s], axis=0)
        #temp = [np.log2(1 + snr * si) for si in s]
        #print (np.shape(np.sum(temp, axis=0)))
    
    my_u, my_s, my_vh = svd(my_ch)    
    for rho in snr:
        print (waterfilling(my_s, rho))
    c_erg_single_mode = c_single_mode/len(channels)
    c_equal_power = c_equal_power/len(channels)
    plt.plot(snr_db, c_erg_single_mode, label='c_erg single mode')
    plt.plot(snr_db, c_equal_power, label='c_erg equal power')
    plt.legend()
    plt.grid()
    plt.show()
