from utils import katsavounidis_initial_codebook_2, get_beamforming_vector_from_sample
import sys
import numpy as np


if __name__ == "__main__":
    pass
    samples_pathfile = sys.argv[1]
    print (samples_pathfile)
    samples = np.load(samples_pathfile)
    print (np.shape(samples))
    num_of_samples = 6
    samples = np.array([get_beamforming_vector_from_sample(samples[i]) for i in range(num_of_samples)])
    print (np.shape(samples))
    katsavounidis_initial_codebook_2(samples, 5)

