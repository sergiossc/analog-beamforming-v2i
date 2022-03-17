from numpy.linalg import svd
import numpy as np
import pandas as pd
#from plot import AntennaPartern

def samplecodebook2beampatterncodebook(codebook, codebook_name, tx_array_vec):
    """
        Apply Singular Value Decomposition (SVD) to each codeword matrix do get codewords of beampattern. So, it save them in a csv file. It is ready to use on testing time.
        Input: Codebook with K codewords, each of them has the same shape of sample of shannel.
        Output: A csv file with a codebook with K codewords. Each codeword is a analog beamforming. 
    """
    beampatterncodebook = {}
    for cw_id, cw in codebook.items():
        u, d, vh = svd(cw)
        w = vh[0]
        #AntennaPartern(w, tx_array_vec)
        #uf = 1 * np.exp(1j * np.angle(vh.conj())) 
        print ('w: ', w)
        print ('w.shape: ', w.shape)
        #w = {i: w[i] for i in range(len(w))}
        beampatterncodebook[cw_id] = w
    df = pd.DataFrame(data=beampatterncodebook)
    csv_filename = 'tx_codebook_' + codebook_name + '.csv'
    df.to_csv(csv_filename)


#    df_rec = pd.read_csv('tx_codebook_da0bc8b2-a1c4-46c5-9e9a-457bd9ff64fe.csv')
#    codebk = df_rec.to_dict() 
#    for k, v in codebk.items():
#        print ('k: ', k)
#        print ('v: ', v.values())
