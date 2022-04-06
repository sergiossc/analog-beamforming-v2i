import sys
import numpy as np
from numpy.linalg import svd, norm
#from lib.vq.utils import norm
import matplotlib.pyplot as plt
from utils import gen_channel, get_frab, decode_codebook, beamsweeping2
import json


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
   


    # Getting samples of channel
    print(f'Getting some samples of channel...')

    samples_pathfile = sys.argv[1]
    channels = np.load(samples_pathfile) 
    num_samples, n, m = np.shape(channels)

    num_of_trials = 100
    np.random.seed(5678)
    ch_id_list = np.random.choice(num_samples, num_of_trials, replace=False)

    codebook_est_pathfile_json = sys.argv[2]
    with open(codebook_est_pathfile_json) as result:
        result_data = result.read()
        result_d = json.loads(result_data)
    #initial_alphabet_opt = result_d['initial_alphabet_opt']
    cb_dict_est = decode_codebook(result_d['codebook'])
    cb_dict_egt = {}
    cw_id = 0
    for k, v in cb_dict_est.items():
        cw_id += 1
        cb_dict_egt[f'cw{cw_id}'] = 1/np.sqrt(m) * np.exp(1j * np.angle(v))


    # Getting a frab version of cb_dict
    phase_resolution_bits_avaiable = [1, 2, 3, 4]
    cb_dict_frab = {}
    for phase_resolution_bits in phase_resolution_bits_avaiable:
        cb_dict_frab[phase_resolution_bits] = {}
    for phase_resolution_bits in phase_resolution_bits_avaiable:
        for k, v in cb_dict_egt.items():
            #cb_dict_frab[phase_resolution_bits][k] = 1/np.sqrt(n_tx) * np.exp(1j * np.angle(get_frab(v, phase_resolution_bits)))
            cb_dict_frab[phase_resolution_bits][k] = get_frab(v, phase_resolution_bits)
            print (norm(cb_dict_frab[phase_resolution_bits][k]))

    
    snr_db = np.arange(-20, 20, 1)
    snr = 10 ** (snr_db/10)

    bf_gain_opt_egt_mean = []    
    bf_gain_rt_egt_mean = []
    bf_gain_rt_egt_frab_1b_mean = []
    bf_gain_rt_egt_frab_2b_mean = []
    bf_gain_rt_egt_frab_3b_mean = []
    bf_gain_rt_egt_frab_4b_mean = []

    for snr_v in snr:

        bf_gain_opt_egt = []
        bf_gain_rt_egt = []
        bf_gain_rt_egt_frab_b1 = []
        bf_gain_rt_egt_frab_b2 = []
        bf_gain_rt_egt_frab_b3 = []
        bf_gain_rt_egt_frab_b4 = []


        for ch_id in ch_id_list:
            pass
            ch = channels[ch_id]
    
            bf_gain_opt, f_opt, w_opt = gf_gain(ch)
            f_egt = get_egt_version(f_opt, m)
            gain_opt_egt = bf_gain_test(ch, f_egt, w_opt) # opt receiver
            bf_gain_opt_egt.append(np.log2(1 +  snr_v * gain_opt_egt))
            
            gain_est_egt, cw_id_tx = beamsweeping2(ch, cb_dict_egt)
            bf_gain_rt_egt.append( np.log2(1 +  snr_v * gain_est_egt))
            f_egt_est = cb_dict_egt[cw_id_tx]
    
            b = 4
            #gain_est_egt_b4, cw_id_tx_b4 = beamsweeping2(ch, cb_dict_frab[b])
            f_egt_frab_b4 = get_frab(f_egt_est, b)
            gain_est_egt_b4 = bf_gain_test(ch,  f_egt_frab_b4   , w_opt) # opt receiver
            bf_gain_rt_egt_frab_b4.append(np.log2(1 + snr_v *  gain_est_egt_b4))
#    
            b = 3
            #gain_est_egt_b3, cw_id_tx_b3 = beamsweeping2(ch, cb_dict_frab[b])
            f_egt_frab_b3 = get_frab( f_egt_est, b)
            gain_est_egt_b3 = bf_gain_test(ch,  f_egt_frab_b3   , w_opt) # opt receiver
            bf_gain_rt_egt_frab_b3.append(np.log2(1 + snr_v *  gain_est_egt_b3))
#    
            b = 2
            #gain_est_egt_b2, cw_id_tx_b2 = beamsweeping2(ch, cb_dict_frab[b])
            f_egt_frab_b2 = get_frab( f_egt_est, b)
            gain_est_egt_b2 = bf_gain_test(ch,  f_egt_frab_b2   , w_opt) # opt receiver
            bf_gain_rt_egt_frab_b2.append(np.log2(1 + snr_v *  gain_est_egt_b2))
#    
            b = 1
            f_egt_frab_b1 = get_frab( f_egt_est, b)
            gain_est_egt_b1 = bf_gain_test(ch,  f_egt_frab_b1   , w_opt) # opt receiver
            bf_gain_rt_egt_frab_b1.append(np.log2(1 + snr_v *  gain_est_egt_b1))
    
    
        bf_gain_opt_egt_mean.append(np.mean(bf_gain_opt_egt))
        bf_gain_rt_egt_mean.append(np.mean(bf_gain_rt_egt))
        bf_gain_rt_egt_frab_1b_mean.append(np.mean(bf_gain_rt_egt_frab_b1)) 
        bf_gain_rt_egt_frab_2b_mean.append(np.mean(bf_gain_rt_egt_frab_b2)) 
        bf_gain_rt_egt_frab_3b_mean.append(np.mean(bf_gain_rt_egt_frab_b3)) 
        bf_gain_rt_egt_frab_4b_mean.append(np.mean(bf_gain_rt_egt_frab_b4)) 
#
#    
    plt.plot(snr_db, bf_gain_opt_egt_mean, label='BF gain OPT (EGT)')
    plt.plot(snr_db, bf_gain_rt_egt_mean, label='BF gain EST (EGT)')
    #plt.plot(bf_gain_random_opt, label='RANDOM - BF gain (OPT)')
    
    #plt.plot(bf_gain_random_egt, label='RANDOM - BF gain (EGT)', linestyle='--')
    
    plt.plot(snr_db, bf_gain_rt_egt_frab_4b_mean, label='BF gain EST (EGT) - b=4', linestyle='--')
    plt.plot(snr_db, bf_gain_rt_egt_frab_3b_mean, label='BF gain EST (EGT) - b=3', linestyle='--')
    plt.plot(snr_db, bf_gain_rt_egt_frab_2b_mean, label='BF gain EST (EGT) - b=2', linestyle='--')
    plt.plot(snr_db, bf_gain_rt_egt_frab_1b_mean, label='BF gain EST (EGT) - b=1', linestyle='--')
    
    plt.legend(loc='best')
    plt.show()



