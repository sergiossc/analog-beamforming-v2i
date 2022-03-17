from utils import sorted_samples, get_beamforming_vector_from_sample
import sys
import numpy as np


if __name__ == "__main__":
    pass
    samples_pathfile = sys.argv[1]
    print (samples_pathfile)
    samples = np.load(samples_pathfile)
    num_of_samples = 6
    samples = np.array([get_beamforming_vector_from_sample(samples[i]) for i in range(num_of_samples)])
    print (np.shape(samples))
    sorted_samples, atrr_values = sorted_samples(samples, 'abs_mean_characteristic_value')
    print (np.shape(sorted_samples))
    print (atrr_values)
