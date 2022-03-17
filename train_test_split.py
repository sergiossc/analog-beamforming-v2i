import numpy as np
import sys

if __name__ == '__main__':
    # load data
    npy_datafile = sys.argv[1]
     
    data = np.load(npy_datafile)
    print (f'data.shape: {np.shape(data)}')
    n_samples, n_rx, n_tr = np.shape(data)
    print (f'n_samples: {n_samples}')

    # kfold setup
    training_size = 0.6 # 0.6 is 3/5 of data
    random_state = True

    # randomize
    if random_state:
        data_id_list = np.random.choice(np.arange(n_samples), n_samples, replace=False)
        randomized_data = []
        for data_id in data_id_list:
            randomized_data.append(data[data_id])
        data = np.array(randomized_data)

    print (f'data.shape: {np.shape(data)}')

    # split data
    training_begin = 0
    training_end = int(n_samples * training_size)

    test_begin = training_end
    test_end = n_samples

    training_set = data[training_begin:training_end]
    test_set = data[test_begin:test_end]


    print (f'training_set_{npy_datafile}.shape: {np.shape(training_set)}')
    print (f'test_set_{npy_datafile}: {np.shape(test_set)}')
    
    # saving new datasets
    np.save(f'training_set_{npy_datafile}', training_set)
    np.save(f'test_set_{npy_datafile}', test_set)
    print ('hello')    
