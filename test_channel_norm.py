import sys
import numpy as np
from numpy.linalg import svd, norm
#from lib.vq.utils import norm
import matplotlib.pyplot as plt


if __name__ == "__main__":
   
    samples_pathfile = sys.argv[1]
    
    # getting samples of channel
    print(f'Getting some samples of channel from {samples_pathfile}...')
    
    channels = np.load(samples_pathfile)
#    ch_id = np.random.choice(len(channels)) 
#    my_ch = channels[ch_id]
#    my_ch = 4 * my_ch
#    #print ('channels.shape: ', channels.shape)
#    #print ('ch.shape: ', ch.shape)
#
#    snr_db = np.arange(-20, 20, 0.01)
#    snr = 10 ** (snr_db/10)
#    
#    c_single_mode = np.zeros(len(snr))
#    c_equal_power = np.zeros(len(snr))
    norm_values = []
    channels_normalized = []
    for ch in channels:
        pass
        n, m = np.shape(ch)
        print ((n,m))
        
        ch = np.sqrt(n * m) * ch/norm(ch)
        norm_v = norm(ch)
        norm_values.append(norm_v)
        channels_normalized.append(ch)
#        ch = 4 * ch
#        u, s, vh = svd(ch)
#        s = s ** 2
#        #print (s)
#        c_single_mode = c_single_mode + np.log2(1 + snr * s[0])
#    
#        c_equal_power = c_equal_power + np.sum([np.log2(1 + snr * si) for si in s], axis=0)
#        #temp = [np.log2(1 + snr * si) for si in s]
#        #print (np.shape(np.sum(temp, axis=0)))
#    
#    my_u, my_s, my_vh = svd(my_ch)    
#    for rho in snr:
#        print (waterfilling(my_s, rho))
#    c_erg_single_mode = c_single_mode/len(channels)
#    c_equal_power = c_equal_power/len(channels)
    plt.plot(norm_values)
    plt.show()
    samples_filename_normalized = f'{samples_pathfile[0:-4]}_normalized.npy'
    np.save(samples_filename_normalized, np.array(channels_normalized))
    print (samples_filename_normalized)
