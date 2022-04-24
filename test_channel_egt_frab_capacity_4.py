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
   
    samples_pathfile = None
    try:
        samples_pathfile = sys.argv[1]
    except:
        print (f'samples_pathfile is {samples_pathfile}')
        exit()

    codebook_est_pathfile_json = None
    try:
        codebook_est_pathfile_json = sys.argv[2]
    except:
        print (f'codebook_est_pathfile_json is {codebook_est_pathfile_json}. \nAnalizing only the samples... ')
        #exit()


    channels = np.load(samples_pathfile) 
    num_samples, n, m = np.shape(channels)

    num_of_trials = 1000
    #np.random.seed(5678)
    #np.random.seed(78)
    ch_id_list = np.random.choice(num_samples, num_of_trials, replace=False)

    if codebook_est_pathfile_json is not None:
        with open(codebook_est_pathfile_json) as result:
            result_data = result.read()
            result_d = json.loads(result_data)
        initial_alphabet_opt = result_d['initial_alphabet_opt']
        cb_est_dict = decode_codebook(result_d['codebook'])
        cb_est_egt_dict = {}
        cw_id = 0
        for k, v in cb_est_dict.items():
            cw_id += 1
            cb_est_egt_dict[cw_id] = (1/np.sqrt(m)) * np.exp(1j * np.angle(v))


    cb_est_egt_dict_b10 = {}
    cb_est_egt_dict_b8 = {}
    cb_est_egt_dict_b4 = {}
    cb_est_egt_dict_b3 = {}
    cb_est_egt_dict_b2 = {}
    cb_est_egt_dict_b1 = {}

    for k, v in cb_est_dict.items():
        cb_est_egt_dict_b10[k] = get_frab(v, 10)
        cb_est_egt_dict_b8[k] = get_frab(v, 8)
        cb_est_egt_dict_b4[k] = get_frab(v, 4)
        cb_est_egt_dict_b3[k] = get_frab(v, 3)
        cb_est_egt_dict_b2[k] = get_frab(v, 2)
        cb_est_egt_dict_b1[k] = get_frab(v, 1)

    snr_db = np.arange(-20, 21, 1)
    snr = 10 ** (snr_db/10)

    capacity_opt_egt_mean = []    
    capacity_opt_egt_b10_mean = []    
    capacity_opt_egt_b8_mean = []    
    capacity_opt_egt_b4_mean = []    
    capacity_opt_egt_b3_mean = []    
    capacity_opt_egt_b2_mean = []    
    capacity_opt_egt_b1_mean = []    

    capacity_est_egt_mean = []    
    capacity_est_egt_b10_mean = []    
    capacity_est_egt_b8_mean = []    
    capacity_est_egt_b4_mean = []    
    capacity_est_egt_b3_mean = []    
    capacity_est_egt_b2_mean = []    
    capacity_est_egt_b1_mean = []    

    for snr_v in snr:
        pass
        ##print (snr_v)
        capacity_opt_egt = []    
        capacity_opt_egt_b10 = []    
        capacity_opt_egt_b8 = []    
        capacity_opt_egt_b4 = []    
        capacity_opt_egt_b3 = []    
        capacity_opt_egt_b2 = []    
        capacity_opt_egt_b1 = []    

        capacity_est_egt = []    
        capacity_est_egt_b10 = []    
        capacity_est_egt_b8 = []    
        capacity_est_egt_b4 = []    
        capacity_est_egt_b3 = []    
        capacity_est_egt_b2 = []    
        capacity_est_egt_b1 = []    

        for ch_id in ch_id_list:
            pass
            ##print (f'ch_id: {ch_id}')
            ch = channels[ch_id]
    
            #OPT
            bf_gain_opt, f_opt, w_opt = gf_gain(ch)
            f_opt_egt = (1/np.sqrt(m)) * np.exp(1j * np.angle(f_opt))
            gain_opt_egt = bf_gain_test(ch, f_opt_egt, w_opt) # opt receiver
            capacity_opt_egt.append(np.log2(1 +  snr_v * gain_opt_egt))


            f_opt_egt_b10 = get_frab(f_opt_egt, 10)
            gain_opt_egt_b10 = bf_gain_test(ch, f_opt_egt_b10, w_opt)
            capacity_opt_egt_b10.append(np.log2(1 + snr_v *  gain_opt_egt_b10))
 


            f_opt_egt_b8 = get_frab(f_opt_egt, 8)
            gain_opt_egt_b8 = bf_gain_test(ch, f_opt_egt_b8, w_opt)
            capacity_opt_egt_b8.append(np.log2(1 + snr_v *  gain_opt_egt_b8))
 
            #gain_opt_egt_b4, cw_id_tx_b4 = beamsweeping2(ch, cb_dict_frab_b4)

            f_opt_egt_b4 = get_frab(f_opt_egt, 4)
            gain_opt_egt_b4 = bf_gain_test(ch, f_opt_egt_b4, w_opt)
            capacity_opt_egt_b4.append(np.log2(1 + snr_v *  gain_opt_egt_b4))
    
            f_opt_egt_b3 = get_frab(f_opt_egt, 3)
            gain_opt_egt_b3 = bf_gain_test(ch, f_opt_egt_b3, w_opt)
            capacity_opt_egt_b3.append(np.log2(1 + snr_v *  gain_opt_egt_b3))
 
            f_opt_egt_b2 = get_frab(f_opt_egt, 2)
            gain_opt_egt_b2 = bf_gain_test(ch, f_opt_egt_b2, w_opt)
            capacity_opt_egt_b2.append(np.log2(1 + snr_v *  gain_opt_egt_b2))
 
            f_opt_egt_b1 = get_frab(f_opt_egt, 1)
            gain_opt_egt_b1 = bf_gain_test(ch, f_opt_egt_b1, w_opt)
            capacity_opt_egt_b1.append(np.log2(1 + snr_v *  gain_opt_egt_b1))
#




            
            #EST
            gain_est_egt, cw_id_tx = beamsweeping2(ch, cb_est_egt_dict)
            capacity_est_egt.append(np.log2(1 + snr_v * gain_est_egt))
            f_est_egt = cb_est_egt_dict[cw_id_tx]
            ##print (f'norm of {norm(f_est_egt)}')
    
            #gain_est_egt_b4, cw_id_tx_b4 = beamsweeping2(ch, cb_dict_frab_b4)

            f_est_egt_b10 = get_frab(f_est_egt, 10)
            gain_est_egt_b10 = bf_gain_test(ch, f_est_egt_b10, w_opt)
            capacity_est_egt_b10.append(np.log2(1 + snr_v *  gain_est_egt_b10))
 

            f_est_egt_b8 = get_frab(f_est_egt, 8)
            gain_est_egt_b8 = bf_gain_test(ch, f_est_egt_b8, w_opt)
            capacity_est_egt_b8.append(np.log2(1 + snr_v *  gain_est_egt_b8))
 


            f_est_egt_b4 = get_frab(f_est_egt, 4)
            gain_est_egt_b4 = bf_gain_test(ch, f_est_egt_b4, w_opt)
            capacity_est_egt_b4.append(np.log2(1 + snr_v *  gain_est_egt_b4))
    
            f_est_egt_b3 = get_frab(f_est_egt, 3)
            gain_est_egt_b3 = bf_gain_test(ch, f_est_egt_b3, w_opt)
            capacity_est_egt_b3.append(np.log2(1 + snr_v *  gain_est_egt_b3))
 
            f_est_egt_b2 = get_frab(f_est_egt, 2)
            gain_est_egt_b2 = bf_gain_test(ch, f_est_egt_b2, w_opt)
            capacity_est_egt_b2.append(np.log2(1 + snr_v *  gain_est_egt_b2))
 
            f_est_egt_b1 = get_frab(f_est_egt, 1)
            gain_est_egt_b1 = bf_gain_test(ch, f_est_egt_b1, w_opt)
            capacity_est_egt_b1.append(np.log2(1 + snr_v *  gain_est_egt_b1))
#    
        capacity_opt_egt_mean.append(np.mean( capacity_opt_egt ))
        capacity_opt_egt_b10_mean.append(np.mean( capacity_opt_egt_b10 ))
        capacity_opt_egt_b8_mean.append(np.mean( capacity_opt_egt_b8 ))
        capacity_opt_egt_b4_mean.append(np.mean( capacity_opt_egt_b4 ))
        capacity_opt_egt_b3_mean.append(np.mean( capacity_opt_egt_b3 ))
        capacity_opt_egt_b2_mean.append(np.mean( capacity_opt_egt_b2 ))
        capacity_opt_egt_b1_mean.append(np.mean( capacity_opt_egt_b1 ))

        capacity_est_egt_mean.append(np.mean( capacity_est_egt ))
        capacity_est_egt_b10_mean.append(np.mean( capacity_est_egt_b10 ))
        capacity_est_egt_b8_mean.append(np.mean( capacity_est_egt_b8 ))
        capacity_est_egt_b4_mean.append(np.mean( capacity_est_egt_b4 ))
        capacity_est_egt_b3_mean.append(np.mean( capacity_est_egt_b3 ))
        capacity_est_egt_b2_mean.append(np.mean( capacity_est_egt_b2 ))
        capacity_est_egt_b1_mean.append(np.mean( capacity_est_egt_b1 ))


    data = {}

    data['snr_db'] = snr_db.tolist()

    data['capacity_opt_egt_mean'] = capacity_opt_egt_mean
    data['capacity_opt_egt_b10_mean'] = capacity_opt_egt_b10_mean
    data['capacity_opt_egt_b8_mean'] = capacity_opt_egt_b8_mean
    data['capacity_opt_egt_b4_mean'] = capacity_opt_egt_b4_mean
    data['capacity_opt_egt_b3_mean'] = capacity_opt_egt_b3_mean
    data['capacity_opt_egt_b2_mean'] = capacity_opt_egt_b2_mean
    data['capacity_opt_egt_b1_mean'] = capacity_opt_egt_b1_mean

    data['capacity_est_egt_mean'] = capacity_est_egt_mean
    data['capacity_est_egt_b10_mean'] = capacity_est_egt_b10_mean
    data['capacity_est_egt_b8_mean'] = capacity_est_egt_b8_mean
    data['capacity_est_egt_b4_mean'] = capacity_est_egt_b4_mean
    data['capacity_est_egt_b3_mean'] = capacity_est_egt_b3_mean
    data['capacity_est_egt_b2_mean'] = capacity_est_egt_b2_mean
    data['capacity_est_egt_b1_mean'] = capacity_est_egt_b1_mean


    json_filename = f'capacity_mean_{initial_alphabet_opt}_{n}x{m}.json'
    with open(json_filename, "w") as write_file:
        json.dump(data, write_file, indent=4)



    fig, ax = plt.subplots(figsize=(6,6))

    ##ax.plot(snr_db, capacity_opt_egt_mean, label=f'ideal (EGT)')
    ##ax.plot(snr_db, capacity_opt_egt_b10_mean, label=f'ideal (EGT) - 10 bits', linestyle='dotted')
    ##ax.plot(snr_db, capacity_opt_egt_b8_mean, label=f'ideal (EGT) - 8 bits', linestyle='dotted')
    ##ax.plot(snr_db, capacity_opt_egt_b4_mean, label=f'ideal (EGT) - 4 bits', linestyle='dotted')
    ##ax.plot(snr_db, capacity_opt_egt_b3_mean, label=f'ideal (EGT) - 3 bits', linestyle='dashed')
    ##ax.plot(snr_db, capacity_opt_egt_b2_mean, label=f'ideal (EGT) - 2 bits', linestyle='dashdot')
    ##ax.plot(snr_db, capacity_opt_egt_b1_mean, label=f'ideal (EGT) - 1 bits', linestyle=(0,(1,10)))
 


    ax.plot(snr_db, capacity_est_egt_mean, label=f'{initial_alphabet_opt} (EGT)')
    ax.plot(snr_db, capacity_est_egt_b10_mean, label=f'{initial_alphabet_opt} (EGT) - 10 bits', linestyle='--')
    ax.plot(snr_db, capacity_est_egt_b8_mean, label=f'{initial_alphabet_opt} (EGT) - 8 bits', linestyle='--')
    ax.plot(snr_db, capacity_est_egt_b4_mean, label=f'{initial_alphabet_opt} (EGT) - 4 bits', linestyle='--')
    ax.plot(snr_db, capacity_est_egt_b3_mean, label=f'{initial_alphabet_opt} (EGT) - 3 bits', linestyle='--')
    ax.plot(snr_db, capacity_est_egt_b2_mean, label=f'{initial_alphabet_opt} (EGT) - 2 bits', linestyle='--')
    ax.plot(snr_db, capacity_est_egt_b1_mean, label=f'{initial_alphabet_opt} (EGT) - 1 bits', linestyle='--')
    ax.grid()
    ax.legend(loc='best')

    ax_small = fig.add_axes([0.60, 0.2, 0.2, 0.2])

    ##ax_small.plot(snr_db, capacity_opt_egt_mean, label=f'ideal (EGT)')
    ##ax_small.plot(snr_db, capacity_opt_egt_b10_mean, label=f'ideal (EGT) - 10 bits', linestyle='dotted')
    ##ax_small.plot(snr_db, capacity_opt_egt_b8_mean, label=f'ideal (EGT) - 8 bits', linestyle='dotted')
    ##ax_small.plot(snr_db, capacity_opt_egt_b4_mean, label=f'ideal (EGT) - 4 bits', linestyle='dotted')
    ##ax_small.plot(snr_db, capacity_opt_egt_b3_mean, label=f'ideal (EGT) - 3 bits', linestyle='dashed')
    ##ax_small.plot(snr_db, capacity_opt_egt_b2_mean, label=f'ideal (EGT) - 2 bits', linestyle='dashdot')
    ##ax_small.plot(snr_db, capacity_opt_egt_b1_mean, label=f'ideal (EGT) - 1 bits', linestyle=(0,(1,10)))
 


    ax_small.plot(snr_db, capacity_est_egt_mean, label=f'{initial_alphabet_opt} (EGT)')
    ax_small.plot(snr_db, capacity_est_egt_b10_mean, label=f'{initial_alphabet_opt} (EGT) - 10 bits', linestyle='--')
    ax_small.plot(snr_db, capacity_est_egt_b8_mean, label=f'{initial_alphabet_opt} (EGT) - 8 bits', linestyle='--')
    ax_small.plot(snr_db, capacity_est_egt_b4_mean, label=f'{initial_alphabet_opt} (EGT) - 4 bits', linestyle='--')
    ax_small.plot(snr_db, capacity_est_egt_b3_mean, label=f'{initial_alphabet_opt} (EGT) - 3 bits', linestyle='--')
    ax_small.plot(snr_db, capacity_est_egt_b2_mean, label=f'{initial_alphabet_opt} (EGT) - 2 bits', linestyle='--')
    ax_small.plot(snr_db, capacity_est_egt_b1_mean, label=f'{initial_alphabet_opt} (EGT) - 1 bits', linestyle='--')

    #ax_small.plot(snr_db, bf_gain_opt_egt_mean, label='ideal (EGT)')
    #ax_small.plot(snr_db, bf_gain_rt_egt_mean, label='xiaoxiao (EGT)')
    #ax_small.plot(snr_db, bf_gain_rt_egt_frab_4b_mean, label='xiaoxiao (EGT) - 4 bits', linestyle='--')
    #ax_small.plot(snr_db, bf_gain_rt_egt_frab_3b_mean, label='xiaoxiao (EGT) - 3 bits' , linestyle='--')
    #ax_small.plot(snr_db, bf_gain_rt_egt_frab_2b_mean, label='xiaoxiao (EGT) - 2 bits', linestyle='--')
    #ax_small.plot(snr_db, bf_gain_rt_egt_frab_1b_mean, label='xiaoxiao (EGT) - 1 bit', linestyle='--')
    
    ax_small.grid()
    ax_small.set_xlim(-1.25, 0.25)
    ax_small.set_ylim(3.66, 4.1)


    fig.text(0.5, 0.05, 'SNR (dB)', ha='center')
    fig.text(0.06, 0.5, r'Capacidade (bps/Hz)', va='center', rotation='vertical')

    
    #plt.xlabel('SNR(dB)')
    #plt.ylabel('Capacidade (bps/Hz)')
#
    plt.show()
    image_filename = f'test-{n}x{m}-{initial_alphabet_opt}_new.png'
    #plt.savefig(image_filename, bbox_inches='tight')
    print (image_filename)
#
