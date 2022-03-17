from utils import get_codebook, get_frab, plot_beamforming_from_codeword, fftmatrix
import sys
import numpy as np
from numpy.linalg import norm

if __name__ == "__main__":
    pass
    json_pathfile = str(sys.argv[1])
    print (json_pathfile)
    cb = get_codebook(json_pathfile)
    #print(cb)
    fftmat, ifftmat = fftmatrix(4, 2)
    nrows, ncols = np.shape(fftmat)
    for k, v in cb.items():
    #for col in range(ncols):
        pass 
        norm_v = norm(v)
        print (f'norm v: {norm(v)}')
        #print (norm_v)
        v_frab = get_frab(v, 1)
        v_frab = 1/np.sqrt(4) * np.exp(1j * np.angle(v_frab))
        print (f'norm v_frab: {norm(v_frab)}')
        print (f'v.abs: \n{np.abs(v)}')
        print (f'v.angle: \n{np.rad2deg(np.angle(v))}')
        print (f'v_frab.abs: \n{np.abs(v_frab)}')
        print (f'v_frab.angle: \n{np.rad2deg(np.angle(v_frab))}')
        
        plot_beamforming_from_codeword(np.column_stack((v, v_frab)))
