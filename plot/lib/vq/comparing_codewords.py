import json
import numpy as np
import sys
from utils import *

if __name__ == '__main__':

    pathfile = sys.argv[1]
        
    with open(pathfile) as result:
        data = result.read()
        d = json.loads(data)

    lloydcodebook_dict = decode_codebook(d['lloydcodebook'])
    dftcodebook_dict = decode_codebook(d['dftcodebook'])


    sets = {}
    for k in lloydcodebook_dict.keys():
        sets[k] = []

    print ('len(lloydcd): ', len(lloydcodebook_dict))
    print ('len(dftcd): ', len(dftcodebook_dict))

    for lloyd_cw_id, lloyd_cw in lloydcodebook_dict.items():
        print ('XXXXXXXXXXXXXXXXXXXXXXX')
        print ('>>> lloyd_cw_id: ', lloyd_cw_id)
        print ('> lloyd_cw: \n', np.exp(1j * np.angle(lloyd_cw)))
        max_correlation = -np.Inf
        dft_cw_id_max_correlation = None
        for dft_cw_id, dft_cw in dftcodebook_dict.items():
            correlation = complex_correlation(lloyd_cw, dft_cw)
            if correlation > max_correlation:
                max_correlation = correlation
                dft_cw_id_max_correlation = dft_cw_id
        print ('>>> dft_cw_id_max_correlation: ', dft_cw_id_max_correlation)
        print ('> dft_cw_max_correlation: \n', dftcodebook_dict[dft_cw_id_max_correlation])
        print ('>>> correlation between lloyd_cw and dft_cw: ', max_correlation)
        sets[lloyd_cw_id].append(dft_cw_id_max_correlation)

        


