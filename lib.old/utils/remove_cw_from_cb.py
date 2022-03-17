import numpy as np
import sys

def remove_cw_from_cb(codebook_filename):    
    #codebook_filename = 'lloyd.npy'
    print ('This is a python code to select keys from a dict in a file.\n')
    print ('Codebook filename: ', codebook_filename)
    print ('In this codebook file there are the following keys:\n')
    
    codebook_dict = np.load(codebook_filename,allow_pickle='TRUE').item()
    new_codebook_dict = np.load(codebook_filename,allow_pickle='TRUE').item()
    for k, v in codebook_dict.items():
        print ('key: ', k)
    
    
    while True:
        key = input('Chose a key to remove from codebook (type \'exit\' to exit): ')
        if key == 'exit':
            print ('Bye.')
            break
        else:
            #print ('k:', key)
            #key = '4ad98e78-4087-4d16-aaf9-ea932603b77b'
            #filename = 'lloyd.npy'
            #lloyd_dictionary = np.load(filename,allow_pickle='TRUE').item()
            found = False
            for k, v in codebook_dict.items():
                if str(k) == key:
                    found = True
                    print ('Key removed.')
                    new_codebook_dict.pop(k)
                    print ('New codebbok size: ', len(new_codebook_dict))
            if found is False:
                print ('Key NOT found!')        
            #lloyd_dictionary.pop(key)
            #np.save(lloyd_dictionary, 'lloyd-mod.npy')
    if len(codebook_dict) != len(new_codebook_dict):
        new_codebook_filename = 'modified_'+codebook_filename
        np.save(new_codebook_filename, new_codebook_dict)
        print ('New codebook file saved in ' + new_codebook_filename)
    else:
        print ('Nothing to do.')

