import numpy as np
from numpy.linalg import norm
from utils import array_to_angular_domain, gen_channel
import sys
import uuid


if __name__ == "__main__":

    nr = int (sys.argv[1])
    nt = int (sys.argv[2])
    num_of_samples = int (sys.argv[3])
 
    kfold_number = 5 


    create_validation_set = True


    dict_channels = {}
    for n in range(num_of_samples):
        k = str(uuid.uuid4())
        v = gen_channel(nr, nt, 1)
        v = v/norm(v)
        dict_channels[k] = v

    #  If angularize is True, get samples in angular domain
    angularize = True
    dict_channels_a = {}
    if angularize:
        for k, v in dict_channels.items():
            h_a = array_to_angular_domain(v)
            h_a = np.sqrt(nr * nt) * h_a/norm(h_a)
            dict_channels_a[k] = h_a
        dict_channels = dict_channels_a


    print (f'%Getting k-fold sets(training, validation, and test), with k = {kfold_number}')
    channels_keys = [k for k in dict_channels.keys()] #[k for k in dict_channels.keys()]
    kfold_slot_length = int(len(channels_keys)/kfold_number)
    if create_validation_set:
        validation_channels_keys = np.random.choice(channels_keys, kfold_slot_length, replace=False)

        validation_channels = {}
        for k in validation_channels_keys:
            validation_channels[k] = dict_channels.pop(k)

    channels_keys = [k for k in dict_channels.keys()]
    test_channels_keys = np.random.choice(channels_keys, kfold_slot_length, replace=False)
    test_channels = {}
    for k in test_channels_keys:
        test_channels[k] = dict_channels.pop(k)

    #training k-fold set
    training_channels = dict_channels
    #
    print (f'...')
    #
    validation_set = []
    validation_set_dict = {}
    
    for k, v in validation_channels.items():
        validation_set.append(v)
    validation_set = np.array(validation_set)
    
    test_set = []
    for k, v in test_channels.items():
        test_set.append(v)
    test_set = np.array(test_set)


    training_set = []
    for k, v in training_channels.items():
        training_set.append(v)
    training_set = np.array(training_set)

    print (f'%training_set.shape = {training_set.shape}')
    print (f'%validation_set.shape = {validation_set.shape}')
    print (f'%test_set.shape = {test_set.shape}')

    angular_str = ''
    if angularize:
        angular_str = '_a'

    tranining_set_filename = f'random-training_set_{nr}x{nt}{angular_str}'
    validation_set_filename = f'random-validation_set_{nr}x{nt}{angular_str}'
    test_set_filename = f'random-test_set_{nr}x{nt}{angular_str}'

    np.save(tranining_set_filename, training_set)
    np.save(validation_set_filename, validation_set)
    np.save(test_set_filename, test_set)

    print (f'%Data saved in \'{tranining_set_filename}.npy\', \'{validation_set_filename}.npy\', \'{test_set_filename}.npy\' files.')
