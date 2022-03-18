import numpy as np
import sys
from utils import richscatteringchnmtx, squared_norm
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt



if __name__ == "__main__":
    pass
    plt.rcParams['text.usetex'] = True
    channels_pathfile = str(sys.argv[1]) 
    print (channels_pathfile) 
    channels = np.load(channels_pathfile)
    n_samples, nr, nt = np.shape(channels)



    s0_list = []
    s1_list = []


    p0_list = []
    p1_list = []


    f_norm_list = []
    f_egt_norm_list = []

    for ch in channels:
        u, s, vh = svd(ch)
        s0_list.append(s[0]**2)
        s1_list.append(s[1]**2)

        f = np.matrix(vh[0,:]).T
        f_norm_list.append(norm(f))
        w = np.matrix(u[:,0]).T

        p0 =  norm(w.conj().T * (ch * f.conj())) ** 2
        p0_list.append(p0)

        f_egt = 1/np.sqrt(nt) * np.exp(1j * np.angle(f))
        f_egt_norm_list.append(norm(f_egt))
        p1 =  norm(w.conj().T * (ch * f_egt.conj())) ** 2
        p1_list.append(p1)





    plt.plot(s0_list, label='s0')
    plt.plot(s1_list, label='s1')
    plt.plot(p0_list, label='p0')
    plt.plot(p1_list, label='p1')
    plt.plot(f_norm_list, label='f_norm')
    plt.plot(f_egt_norm_list, label='f_egt_norm')
    plt.legend()
    plt.show()
