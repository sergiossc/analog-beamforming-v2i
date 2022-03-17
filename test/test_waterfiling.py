import numpy as np
from numpy.linalg import svd, matrix_rank
from lib.vq.utils import norm
import matplotlib.pyplot as plt
from utils import richscatteringchnmtx

def wf(s, snr_v):
    """
        wterfilling power allocation
        input: 'snr' value(float), 's' eigenvalues (numpy.array)
        
        https://www.mathworks.com/matlabcentral/fileexchange/3592-water-filling-algorithm
    """
    vec = 1/(snr_v * s)
    p_alloc = np.zeros(len(vec))
    tol = 0.0001
    pcon = 1
    n = len(vec)
    wline = np.min(vec) + pcon/n
    ptot = np.sum([np.max([wline - vec[i],0]) for i in range(len(vec)) ])
    while (np.abs(pcon-ptot) > tol):
        wline += (pcon-ptot)/n
        p_alloc = np.array([np.max([wline - vec[i],0]) for i in range(len(vec)) ])
        ptot = np.sum(p_alloc)
    #print (p_alloc)
    return p_alloc 

def siso_c(snr_v):
    c = np.log2(1 + snr_v)
    return c

def mimo_csir_c(s, snr_v):
    c = 0
    for i in range(len(s)):
        c += np.log2(1 + snr_v * (1/len(s)) * s[i])
    return c


def mimo_csit_eigenbeamforming_c(s, snr_v):
    c = 0
    opt_power_alloc = wf(s, snr_v)
    for i in range(len(s)):
        c += np.log2(1 + snr_v * opt_power_alloc[i] * s[i])
    return c

def mimo_csit_beamforming(s, snr_v):
    c = 0
    #print (f's0: {s[0]}')
    c = np.log2(1 + snr_v * s[0])
    return c

if __name__ == "__main__":
   
    
    # getting samples of channel
    print("Getting some samples of channel...")
    
    #np.random.seed(1234)
    #channels = np.load('/home/snow/github/land/dataset/npy_files_s007/training_set_4x4.npy')
    #title = f'richscattering'
    title = f'S002-NLOS-RX2'
    channels = np.load('s002_rx2_nlos_channels_4x4.npy')
    #channels = np.load('nlos_channels_4x4.npy')

    snr_db = np.arange(-20, 20, 0.01)
    snr = 10 ** (snr_db/10)

    c_siso_all = []
    c_mimo_csir_all = []
    c_mimo_csit_eigenbeamforming_all = []
    c_mimo_csit_beamforming_all = []

    ch_id_list = np.random.choice(len(channels), 100, replace=False)
    counter = 0
 
    for ch_id in ch_id_list:
        ch = channels[ch_id]
        #ch = richscatteringchnmtx(16, 16, 1.0)
        counter += 1
        print (f'{counter} of {len(ch_id_list)}')
        n = np.shape(ch)[0]
        m = np.shape(ch)[1]
        ch = ch/norm(ch)
        ch = m * ch
    
    
        c_siso = []
        c_mimo_csir = []
        c_mimo_csit_eigenbeamforming = []
        c_mimo_csit_beamforming = []
    
        #u, s, vh = svd(ch * ch.conj().T)    
        u, s, vh = svd(ch * ch.conj().T)    # singular values 
        s = s ** 2 #eigenvalues
        #print (f's: {s}')
        #print (matrix_rank(ch))
        #print (np.sum(s))
    
        for snr_v in snr:
            c_siso.append(siso_c(snr_v))
            c_mimo_csir.append(mimo_csir_c(s, snr_v))
            c_mimo_csit_eigenbeamforming.append(mimo_csit_eigenbeamforming_c(s, snr_v))
            c_mimo_csit_beamforming.append(mimo_csit_beamforming(s, snr_v))

        c_siso_all.append(c_siso)
        c_mimo_csir_all.append(c_mimo_csir)
        c_mimo_csit_eigenbeamforming_all.append(c_mimo_csit_eigenbeamforming)
        c_mimo_csit_beamforming_all.append(c_mimo_csit_beamforming)

    #mine = np.mean(c_mimo_csit_eigenbeamforming_all, axis=0)
    #print (np.shape(mine))
    plt.plot(snr_db, np.mean(c_mimo_csit_eigenbeamforming_all, axis=0),'-.', label=f'MIMO with CSIT (Eigenbeamforming)')
    plt.plot(snr_db, np.mean(c_mimo_csir_all, axis=0),'--', label=f'MIMO with CSIR')
    plt.plot(snr_db, np.mean(c_mimo_csit_beamforming_all, axis=0),'-.', label=f'MIMO with CSIT (Beamforming - Single Mode)')
    plt.plot(snr_db, np.mean(c_siso_all, axis=0), ':', label=f'SISO')
    plt.title(f'{title} n, m = [{n, m}]')
    plt.xlabel(f'Signal-to-Noise Ratio (dB)')
    plt.ylabel(f'Bits per channel use')
    plt.grid()
    plt.legend()
    #plt.show()
    plt.savefig(f'{title}-channels-{n}x{m}-plot.png')
