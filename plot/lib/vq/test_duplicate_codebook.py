import numpy as np
import utils as u

cb = u.gen_dftcodebook(4)
perturbation_variance = 1.0

for cw in cb:
    print (f'*****************\n')
    print (f'shape of cw: {np.shape(cw)}\n')
    print (f'norm of cw: {u.norm(cw)}\n')
    cw_shape = np.shape(cw)
    perturbation_vector = np.sqrt(perturbation_variance/2) * (np.random.randn(cw_shape[0], cw_shape[1]) + 1j * np.random.randn(cw_shape[0], cw_shape[1]))
    new_cb = u.duplicate_codebook(cb, perturbation_vector) 
    for new_cw in new_cb:
    	print (f'shape of new_cw: {np.shape(new_cw)}\n')
    	print (f'norm of new_cw: {u.norm(new_cw)}\n')
    
    pass
