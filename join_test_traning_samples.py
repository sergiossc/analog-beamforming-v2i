import numpy as np

if __name__ == "__main__":
    n_set = [4, 8, 16, 32, 64]
    ds_name_set = ['s002']

    for n in n_set:
        for ds_name in ds_name_set:
            pass
            print (n)
            filename_test = f'{ds_name}-test_set_{n}x{n}_a.npy'
            filename_validation = f'{ds_name}-validation_set_{n}x{n}_a.npy'
            # s002-test_set_4x4_a.npy
            # s002-training_set_4x4_a.npy 
            print (filename_test)
            print (filename_validation)
            test_set = np.load(filename_test)
            val_set = np.load(filename_validation)
            print (np.shape(test_set))
            print (np.shape(val_set))
            join_set = np.vstack((test_set, val_set))
            print (np.shape(join_set))
            filename_joined = f'{ds_name}-test_set_{n}x{n}_a_joined.npy'
            np.save(filename_joined, join_set)
            print (filename_joined)
