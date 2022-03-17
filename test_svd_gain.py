import numpy as np
from utils import richscatteringchnmtx, squared_norm
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt

#np.random.seed(1234)

n = 4 # num of receiver antennas
m = 4 # num of transmitter antenas

num_samples = 100

s_sum = []
s_est_sum = []

for sample in range(num_samples):
    print ('----------')
    var = 1.0
    h = richscatteringchnmtx(n, m, var)
    h = m * h/norm(h) # 
    print (f'norm(h): {norm(h)}')
    u, s, vh = svd(h) # perform singular value decomposition
    s = s ** 2
    s_sum.append(s[0])

    p1 = richscatteringchnmtx(n, m, var)
    p1 = np.sqrt(m) * p1/norm(p1)
    c1 = richscatteringchnmtx(n, m, var)
    c1 = np.sqrt(n) * c1/norm(c1)
    #print (f'norm(p1): {norm(p1)}')
    #print (f'norm(c1): {norm(c1)}')

    precoder = np.matrix(vh).conj().T[:,0] #* np.matrix(np.identity(m)[0]).T 
    #precoder = np.matrix(p1)[:,0]
    #precoder = precoder/norm(precoder)
    print (f'norm(precoder): {norm(precoder)}')
    print (f'precoder.shape: {precoder.shape}')
    combining = np.matrix(u).conj().T[0,:] # * np.matrix(np.identity(n)[0])
    #combining = np.matrix(c1)[0,:]
    #combining = combining/norm(combining)
    print (f'norm(combining): {norm(combining)}')
    print (f'combining.shape: {combining.shape}')
    #combining = c1
    #print (f'norm(combining): {norm(combining)}')
    
    #print (f'precoder: \n{precoder}')

    product1 = h * precoder
    #print (f'p1: {product1.shape}')

    product2 = combining * product1
    print (f'product2.shape: {product2.shape}')
    print (f'p2: {product2}')
    #print (f'norm(p2): {norm(product2)}')

    u_est, s_est, vh_est = svd(product2) # perform singular value decomposition
    s_est = s_est ** 2
    #s_est_sum.append(s_est[0])
    s_est_sum.append(np.abs(product2[0,0]) ** 2)



#max_c = np.argmax(c_vec)
#print (max_c)
#print (sv_vec[max_c])

plt.plot(s_sum, label=f'autovalue[0]')
plt.plot(s_est_sum, label=f'est_autovalue[0]')
#plt.plot(snr_db_vec, c_bf, label=f'beamforming')
#plt.plot(snr_db_vec, c_mimo, label=f'mimo [{n}, {m}]')
#plt.plot(snr_db_vec, c_mu, label=f'multplex gain mimo [{n}, {m}]')
#plt.plot(snr_db_vec, c_miso, label=f'miso [{n}, {1}]')
#plt.plot(snr_db_vec, c_simo, label=f'simo [{1}, {m}]')
#plt.plot(np.sum(sv_vec, axis=1), 'r*', label=f'sv_vec')
plt.legend()
#plt.title(f'Capacity vs SNR')
#plt.xlabel(f'SNR(dB)')
#plt.ylabel(f'Capacity(bps/Hz)')
#plt.grid()
plt.show()

#        for ch_id in ch_id_list:
#            ch = channels[ch_id]
#            n = np.shape(ch)[0]
#            m = np.shape(ch)[1]
#            ch = ch/norm(ch)
#            ch = m * ch
#            #ch = a/norm(a)
#            
#            u, s, vh = svd(ch)    # singular values 
#            s = s ** 2 #eigenvalues
#            print ('eigenvalues of channel----')
#            print (s)
#            print (np.sum(s))
#            print ('----')
#            #s = s ** 2 #eigenvalues
#            vh = np.matrix(vh) 
#            u = np.matrix(u) 
#            f = vh[0,:]
#            w = u[:,0]
#        
#            p = w.conj().T * (ch * f.conj().T)
#            #p = np.abs(p)
#            p = np.abs(p.conj() * p)
#        
#            p_real.append(p)
#            
#            prod = ch.conj().T * ch
#            
#            p_est_max, cw_id_tx, cw_id_rx = beamsweeping(ch, cb_dict)
#            p_est.append(p_est_max)
#            cb_hist[(cw_id_tx,cw_id_rx)] += 1
#        p_real = np.array(p_real).reshape(num_of_trials)
#        print (f'mean of real: {np.mean(p_real)}')
#        p_est = np.array(p_est).reshape(num_of_trials)
#        
#        print (f'mean of estmated: {np.mean(p_est)}')
#        print (f'cb_hist: {cb_hist}')
#        print (f'len(cb_hist): {len(cb_hist)}')
#        ##df = pd.DataFrame(cb_hist.values(), index=cb_hist.keys(), columns=[f'{initial_alphabet_opt}'])
#        df[f'{initial_alphabet_opt}'] = cb_hist.values()
#        df.index = cb_hist.keys()
#        print (f'df: {df}')
#    df.plot.bar(subplots=True, legend=True, title=['','','','','',''], layout=(3,2), sharey=True, sharex=False, figsize=(24, 16), fontsize=8, xlabel='codeword pairwise', ylabel='# of samples', grid=True)
#    plt.show()
