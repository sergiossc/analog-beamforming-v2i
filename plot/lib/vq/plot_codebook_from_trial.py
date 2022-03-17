import json
import numpy as np
import sys
from utils import *

if __name__ == '__main__':

    pathfile = sys.argv[1]
        
    with open(pathfile) as result:
        data = result.read()
        d = json.loads(data)

    instance_id = d['instance_id']

    lloydcodebook_dict = decode_codebook(d['lloydcodebook'])
    initialcodebook_dict = decode_codebook(d['initial_codebook'])

    lloydcodebook_matrix = dict2matrix(lloydcodebook_dict)
    initialcodebook_matrix = dict2matrix(initialcodebook_dict)

    lloyd_nrows, lloyd_ncols = lloydcodebook_matrix.shape
    lloydcodebook_matrix = np.array(lloydcodebook_matrix).reshape(lloyd_nrows, 1, lloyd_ncols)
    plot_codebook(lloydcodebook_matrix, 'lloyd_codebook_'+ instance_id  +'.png')

    initial_nrows, initial_ncols = initialcodebook_matrix.shape
    initialcodebook_matrix = np.array(initialcodebook_matrix).reshape(initial_nrows, 1, initial_ncols)
    plot_codebook(initialcodebook_matrix, 'initial_codebook_' + instance_id + '.png')

        


