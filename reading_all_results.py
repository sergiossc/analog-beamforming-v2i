import os
import json
import uuid
from utils import compare_filter, get_all_result_json_pathfiles, get_all_possible_filters

if __name__ == "__main__":

    result_filter_set, result_filter_counter = get_all_possible_filters()
    rootdir = "results"
    result_pathfiles_dict = get_all_result_json_pathfiles(rootdir)
    #result_pathfiles_dict = {}

    count = 0

    print (f'# of JSON result files: {len(result_pathfiles_dict)}')
    #result_count = 0
    counter = 0
    for k, pathfile in result_pathfiles_dict.items():
        
        with open(pathfile) as result:
            data = result.read()
            d = json.loads(data)
        
        pass
        channel_samples_filename = d['channel_samples_filename']
        #print (channel_samples_filename)
        index = channel_samples_filename.find("s00")
        ds_name = channel_samples_filename[index:index+4]

        initial_alphabet_opt = d['initial_alphabet_opt']
        rx_array_size = d['rx_array_size']
        tx_array_size = d['tx_array_size']
        num_of_levels = d['num_of_levels']

        result_filter = {'ds_name':ds_name , 'initial_alphabet_opt':initial_alphabet_opt, 'rx_array_size':rx_array_size, 'tx_array_size':tx_array_size, 'num_of_levels':num_of_levels }
        #my_result_filter = {'ds_name': 's000', 'rx_array_size': 4, 'tx_array_size': 4, 'initial_alphabet_opt': 'random', 'num_of_levels': 4} 
        #print (result_filter)
        for my_filter_id, my_filter in result_filter_set.items():
        
            if compare_filter(my_filter, result_filter): 
                pass 
                result_filter_counter[my_filter_id] += 1
                counter += 1
                #print (my_result_filter)
                #print (result_filter)
                #result_count = result_count + 1
    print (f'# of matched filters on /{rootdir}: {counter}')
    for k, v in result_filter_counter.items():
        if v==0:
            print (k)
            print (result_filter_set[k])
            print (v)
            print (f'----xxx---x--xxx---xxx---')
