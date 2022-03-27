import sys
import numpy as np
from numpy.linalg import svd, norm
#from lib.vq.utils import norm
import matplotlib.pyplot as plt
from utils import gen_channel, get_frab


#get_frab(complex_mat, b)

def gf_gain(ch):
    pass
    u, s, vh = svd(ch)
    vh = np.matrix(vh)
    u = np.matrix(u)
    f = vh[0,:]
    f = f.conj().T
    w = u[:,0]

    bf_gain_eigen_value_0 = s[0] ** 2
    return bf_gain_eigen_value_0, f, w

def bf_gain_test(ch, f, w):
    pass
    gain = np.abs(w.conj().T * (ch * f)) ** 2
    gain = gain[0,0]
    return gain

def get_egt_version(f, num_tx):
    pass
    f_egt = (1/np.sqrt(num_tx)) * np.exp(1j * np.angle(f))
    return f_egt

if __name__ == "__main__":
   
    samples_pathfile = sys.argv[1]
    channels = np.load(samples_pathfile) 
    

    rt_norm_values = []
    random_norm_values = []

    bf_gain_rt_opt = []
    bf_gain_random_opt = []

    bf_gain_rt_egt = []
    bf_gain_random_egt = []

    bf_gain_random_egt_frab_b4 = []
    bf_gain_random_egt_frab_b3 = []
    bf_gain_random_egt_frab_b2 = []
    bf_gain_random_egt_frab_b1 = []


    
    # Getting samples of channel
    print(f'Getting some samples of channel from {samples_pathfile}...')
    channels = np.load(samples_pathfile)

    for ch in channels:
        pass

        n, m = np.shape(ch)

        rt_norm_v = norm(ch)
        rt_norm_values.append(rt_norm_v)
        rt_bf_gain_opt, rt_f_opt, rt_w_opt = gf_gain(ch)
        bf_gain_rt_opt.append(rt_bf_gain_opt)
        
        rt_f_egt = get_egt_version(rt_f_opt, m)
        print (f'rt_f_egt.norm: {norm(rt_f_egt)}')
        rt_bf_gain_egt = bf_gain_test(ch, rt_f_egt, rt_w_opt)
        bf_gain_rt_egt.append(rt_bf_gain_egt)
        

        random_ch = gen_channel(n, m, 1.0)
        random_ch = np.sqrt(n * m) * random_ch/norm(random_ch)
        random_norm_v = norm(random_ch)
        random_norm_values.append(random_norm_v)
        random_bf_gain_opt, random_f_opt, random_w_opt = gf_gain(random_ch)
        bf_gain_random_opt.append(random_bf_gain_opt)

        random_f_egt = get_egt_version(random_f_opt , m)
        print (f'random_f_egt.norm: {norm(random_f_egt)}')
        random_bf_gain_egt = bf_gain_test(random_ch, random_f_egt, random_w_opt)
        bf_gain_random_egt.append(random_bf_gain_egt)
        

        b = 4
        random_f_egt_frab_b4 = get_frab(random_f_egt, b)
        print (f'random_f_egt_frab_b4.norm: {norm(random_f_egt_frab_b4)}')
        random_bf_gain_egt_frab_b4 = bf_gain_test(random_ch, random_f_egt_frab_b4, random_w_opt)
        bf_gain_random_egt_frab_b4.append(random_bf_gain_egt_frab_b4)




        b = 3
        random_f_egt_frab_b3 = get_frab(random_f_egt, b)
        print (f'random_f_egt_frab_b3.norm: {norm(random_f_egt_frab_b3)}')
        random_bf_gain_egt_frab_b3 = bf_gain_test(random_ch, random_f_egt_frab_b3, random_w_opt)
        bf_gain_random_egt_frab_b3.append(random_bf_gain_egt_frab_b3)



        b = 2
        random_f_egt_frab_b2 = get_frab(random_f_egt, b)
        print (f'random_f_egt_frab_b2.norm: {norm(random_f_egt_frab_b2)}')
        random_bf_gain_egt_frab_b2 = bf_gain_test(random_ch, random_f_egt_frab_b2, random_w_opt)
        bf_gain_random_egt_frab_b2.append(random_bf_gain_egt_frab_b2)

        b = 1
        random_f_egt_frab_b1 = get_frab(random_f_egt, b)
        print (f'random_f_egt_frab_b1.norm: {norm(random_f_egt_frab_b1)}')
        random_bf_gain_egt_frab_b1 = bf_gain_test(random_ch, random_f_egt_frab_b1, random_w_opt)
        bf_gain_random_egt_frab_b1.append(random_bf_gain_egt_frab_b1)



    #plt.plot(bf_gain_rt_opt, label='RT - BF gain (OPT)')
    plt.plot(bf_gain_random_opt, label='RANDOM - BF gain (OPT)')

    #plt.plot(bf_gain_rt_egt, label='RT - BF gain (EGT)', linestyle='--')
    plt.plot(bf_gain_random_egt, label='RANDOM - BF gain (EGT)', linestyle='--')

    plt.plot(bf_gain_random_egt_frab_b4, label='RANDOM - BF gain (EGT) -- frab -- b=4', linestyle='--')
    plt.plot(bf_gain_random_egt_frab_b3, label='RANDOM - BF gain (EGT) -- frab -- b=3', linestyle='--')
    plt.plot(bf_gain_random_egt_frab_b2, label='RANDOM - BF gain (EGT) -- frab -- b=2', linestyle='--')
    plt.plot(bf_gain_random_egt_frab_b1, label='RANDOM - BF gain (EGT) -- frab -- b=1', linestyle='--')

    plt.legend(loc='best')
    plt.show()



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
    ##plt.plot(rt_norm_values, label='RT channels norm')
    ##plt.plot(random_norm_values, label='Random channels norm')


