import sys
import numpy as np
from numpy.linalg import svd, norm
#from lib.vq.utils import norm
import matplotlib.pyplot as plt


if __name__ == "__main__":
   
    samples_pathfile1 = sys.argv[1]
    samples_pathfile2 = sys.argv[2]
    
    samples_pathfile_list = []
    samples_pathfile_list.append(samples_pathfile1)
    samples_pathfile_list.append(samples_pathfile2)
    

    channels_joined = []
    norm_values = []
    
    for samples_pathfile in samples_pathfile_list:

        # getting samples of channel
        print(f'Getting some samples of channel from {samples_pathfile}...')
        channels = np.load(samples_pathfile)

        for ch in channels:
            pass
            #n, m = np.shape(ch)
            #print ((n,m))
            #ch = np.sqrt(n * m) * ch/norm(ch)
            norm_v = norm(ch)
            norm_values.append(norm_v)
            channels_joined.append(ch)
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
    samples_filename_joined = f'{samples_pathfile[0:-4]}_joined.npy'
    np.save(samples_filename_joined, np.array(channels_joined))
    print (samples_filename_joined)
