import sys
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from utils import get_beamforming_vector_from_sample, sorted_samples, xiaoxiao_initial_codebook_2, matrix2dict, dict2matrix, sorted_samples_2

if __name__ == "__main__":
    pass
    npy_samples_filename = str(sys.argv[1])
    print (f'npy_samples_filename: {npy_samples_filename}')
    samples = np.load(npy_samples_filename)
    num_samples, num_rx, num_tx = np.shape(samples)
    print (f'np.shape: {num_samples}x{num_rx}x{num_tx}')
    num_of_trials = num_samples
    ch_id_list = np.random.choice(num_samples, num_of_trials, replace=False)

    training_samples = np.array([get_beamforming_vector_from_sample(samples[i]) for i in ch_id_list])
    training_samples_dict = matrix2dict(training_samples)
    
    print (len(training_samples_dict))
    var_training_samples_not_sorted = []
    for k, v in training_samples_dict.items():
        #print (k)
        pass
        sample = v
        sample = sample.reshape(1, 1, num_tx)
        sample_sorted_var, attr_sorted_var = sorted_samples(sample, 'variance_characteristic_value')
        info_dict = {'sample_id': k, 'var_value': attr_sorted_var[0]}
        var_training_samples_not_sorted.append(info_dict)

    pass

    var_training_samples_sorted = sorted_samples_2(training_samples_dict, 'variance_characteristic_value')
#
    initial_codebook_list = xiaoxiao_initial_codebook_2(training_samples_dict, 4)
    count = 0
    for value in var_training_samples_sorted:
        var_training_samples_sorted_k = value['sample_id']
        for cw_dict in initial_codebook_list:
            k = cw_dict['sample_id']
            if k == var_training_samples_sorted_k:
                print (count)
                cw_dict['pos'] = count
        count += 1
    #cb_sorted_var, cb_attr_sorted_var = sorted_samples(initial_codebook, 'variance_characteristic_value') 
    var_values = np.array([v['s_var'] for v in var_training_samples_sorted])
    y_var_values = np.arange(len(var_values)) / len(var_values)

    #df = pd.DataFrame(var_values, 'frequency')
    #df['cumulative'] =  df[]

    fig, axs = plt.subplots()
    axs.plot([v['var_value'] for v in var_training_samples_not_sorted], 'bo', label='Amostras não ordenadas')
    axs.plot([v['s_var'] for v in var_training_samples_sorted], 'ro', label='Amostras ordenadas')
    axs.plot([v['pos'] for v in initial_codebook_list], [v['s_var'] for v in initial_codebook_list], 'go', label='Codebook inicial')

    #axs[1].plot(var_values, y_var_values, 'g-', label='CDF')

    plt.ylabel('Variância característica')
    plt.xlabel('Amostra de canal')
    #axs[0].legend()
    axs.legend()
    fig_filename = f'initial_codebook_xiaoxiao.png'
    plt.savefig(fig_filename, bbox_inches='tight')
    #plt.show()
###
