import uuid
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from utils import *

#def check_files(prefix, episodefiles):
#    pathfiles = {}
#    for ep_file in episodefiles:
#        pathfile = prefix + str('/') + str(ep_file)
#        ep_file_status = False
#        try:
#            current_file = open(pathfile)
#            ep_file_status = True
#            #print("Sucess.")
#        except IOError:
#            print("File not accessible: ", pathfile)
#        finally:
#            current_file.close()
#
#        if ep_file_status:
#            ep_file_id = uuid.uuid4()
#            pathfiles[ep_file_id] = pathfile
# 
#    return pathfiles
#
#
#def decode_mean_distortion(mean_distortion_dict):
#    mean_distortion_list = []
#    for iteration, mean_distortion in mean_distortion_dict.items():
#        mean_distortion_list.append(mean_distortion)
#    return mean_distortion_list


if __name__ == '__main__':

    profile_pathfile = 'profile.json' 

    with open(profile_pathfile) as profile:
        data = profile.read()
        d = json.loads(data)

    prefix_pathfiles = d['results_directory']
    result_files = os.listdir(prefix_pathfiles)
    pathfiles = check_files(prefix_pathfiles, result_files)
    print ('# of json files: ', len(pathfiles))
    # From here it is going to open each json file to see each parameters and data from algorithm perform. May you should to implement some decode or transate functions to deal with json data from files to python data format. There are some decode functions on utils library. 
    #trial_result = (initial_alphabet_opt, distortion_measure_opt, num_of_levels, variance_of_samples, norm)

    occurences = []
    samples_random_seeds = {}

    katsavounidis_results = {}
    xiaoxiao_results = {}
    unitary_until_num_of_elements_results = {}
    random_from_samples_results = {}
    sa_results = {}

    for pathfile_id, pathfile in pathfiles.items():
        with open(pathfile) as result:
            data = result.read()
            d = json.loads(data)

        # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
        initial_alphabet_opt = d['initial_alphabet_opt']
        variance_of_samples = d['variance_of_samples']
        distortion_measure_opt = d['distortion_measure_opt']
        initial_alphabet_opt = d['initial_alphabet_opt']
        num_of_elements = d['num_of_elements']
        num_of_levels = num_of_elements 
        num_of_samples = d['num_of_samples']
        samples_random_seed = d['samples_random_seed']
        mean_distortion_by_round = d['mean_distortion_by_round']

        #normal_vector = np.ones(num_of_levels) * (num_of_samples/num_of_levels)
        #sets = d['sets']
        #set_vector = []
        #for k, v in sets.items():
        #    set_vector.append(v)
        #set_vector = np.array(set_vector)
   
        #norm =  np.sqrt(np.sum(np.power(np.abs(set_vector - normal_vector), 2)))
        #if norm == 0 and num_of_elements == 9 and variance_of_samples == 1.0 and initial_alphabet_method == 'katsavounidis': 
        #if  norm == 0 and num_of_elements == 4 and variance_of_samples == 0.1 and initial_alphabet_method == 'katsavounidis'
        #if  variance_of_samples == 0.1 and num_of_elements == 4 and initial_alphabet_opt == 'katsavounidis':
            #trial_info = {'norm': norm}
            #occurences.append(trial_info)
        #if  num_of_elements == 4:
        samples_random_seeds[int(samples_random_seed)] = 1

        if initial_alphabet_opt == 'katsavounidis' and distortion_measure_opt == 'mse':
            last_k = ''
            for k in mean_distortion_by_round.keys():
                last_k = k
            mean_distortion_by_round_list = decode_mean_distortion(mean_distortion_by_round[last_k])
            katsavounidis_results[str(int(samples_random_seed))] = mean_distortion_by_round_list[-1] 

        if initial_alphabet_opt == 'xiaoxiao' and distortion_measure_opt == 'mse':
            last_k = ''
            for k in mean_distortion_by_round.keys():
                last_k = k
            mean_distortion_by_round_list = decode_mean_distortion(mean_distortion_by_round[last_k])
            xiaoxiao_results[str(int(samples_random_seed))] = mean_distortion_by_round_list[-1] 

        if initial_alphabet_opt == 'sa' and distortion_measure_opt == 'mse':
            last_k = ''
            for k in mean_distortion_by_round.keys():
                last_k = k
            mean_distortion_by_round_list = decode_mean_distortion(mean_distortion_by_round[last_k])
            sa_results[str(int(samples_random_seed))] = mean_distortion_by_round_list[-1] 


        if initial_alphabet_opt == 'unitary_until_num_of_elements' and distortion_measure_opt == 'mse':
            last_k = ''
            for k in mean_distortion_by_round.keys():
                last_k = k
            mean_distortion_by_round_list = decode_mean_distortion(mean_distortion_by_round[last_k])
            unitary_until_num_of_elements_results[str(int(samples_random_seed))] = mean_distortion_by_round_list[-1] 


        if initial_alphabet_opt == 'random_from_samples' and distortion_measure_opt == 'mse':
            last_k = ''
            for k in mean_distortion_by_round.keys():
                last_k = k
            mean_distortion_by_round_list = decode_mean_distortion(mean_distortion_by_round[last_k])
            random_from_samples_results[str(int(samples_random_seed))] = mean_distortion_by_round_list[-1] 


        occurences.append(1)

    print('occurences.len: ', len(occurences))

    samples_random_seeds = samples_random_seeds.items()
    samples_random_seeds_k = np.array([str(k[0]) for k in sorted(samples_random_seeds)])

    labels = samples_random_seeds_k

    interval_list = []

    # 't' distribution. Degrees of freedom. For a sample of size n, the t distribution will have n-1 degrees of freedom.  As the sample size n increases, the t distribution becomes closer to the normal distribution, since the standard error approaches the true standard deviation for large n. 
    t = 1.699 # 95% confidence, 29 degrees of fredom

    # ---------------------------katsavounivis--------------------------------------
    katsavounidis_results = katsavounidis_results.items()
    print ('len(katsavounidis_results): ', len(katsavounidis_results))
    katsavounidis_v = np.array([float(v[1]) for v in sorted(katsavounidis_results)])
    interval_list.append(get_confidence_interval(katsavounidis_v, t))

    #---------------------------xiaoxiao-----------------------------------------------
    xiaoxiao_results = xiaoxiao_results.items()
    print ('len(xiaoxio_results): ', len(xiaoxiao_results))
    xiaoxiao_v = np.array([float(v[1]) for v in sorted(xiaoxiao_results)])
    interval_list.append(get_confidence_interval(xiaoxiao_v, t))

    #---------------------------sa-----------------------------------------------
    sa_results = sa_results.items()
    print ('len(sa_results): ', len(sa_results))
    sa_v = np.array([float(v[1]) for v in sorted(sa_results)])
    interval_list.append(get_confidence_interval(sa_v, t))

    #---------------------------unitary_until_num_of_elements-----------------------------------------------
    unitary_until_num_of_elements_results = unitary_until_num_of_elements_results.items()
    print ('len(unitary_until_num_of_elements_results): ', len(unitary_until_num_of_elements_results))
    unitary_until_num_of_elements_v = np.array([float(v[1]) for v in sorted(unitary_until_num_of_elements_results)])
    interval_list.append(get_confidence_interval(unitary_until_num_of_elements_v, t))

    #---------------------------random_from_samples-----------------------------------------------
    random_from_samples_results = random_from_samples_results.items()
    print ('len(random_from_samples_results): ', len(random_from_samples_results))
    random_from_samples_v = np.array([float(v[1]) for v in sorted(random_from_samples_results)])
    interval_list.append(get_confidence_interval(random_from_samples_v, t))

    #----------------------------- Percentile stuff ---------------------------------------------------------

    #whis_value = 1.57
    #katsavounidis_1st_percentile, katsavounidis_median, katsavounidis_3rd_percentile, katsavounidis_iqr = get_percentiles(katsavounidis_v)
    #xiaoxiao_1st_percentile, xiaoxiao_median, xiaoxiao_3rd_percentile, xiaoxiao_iqr = get_percentiles(xiaoxiao_v)
    #sa_1st_percentile, sa_median, sa_3rd_percentile, sa_iqr = get_percentiles(sa_v)
    #unitary_until_num_of_elements_1st_percentile, unitary_until_num_of_elements_median, unitary_until_num_of_elements_3rd_percentile, unitary_until_num_of_elements_iqr = get_percentiles(unitary_until_num_of_elements_v)
    #random_from_samples_1st_percentile, random_from_samples_median, random_from_samples_3rd_percentile, random_from_samples_iqr = get_percentiles(random_from_samples_v)
    
    fig, ax = plt.subplots()
    interval_list = np.array(interval_list) 
    print (interval_list[:,0]) 
    print (interval_list[:,1]) 
    print (interval_list[:,2]) 
    err = interval_list[:,2] - interval_list[:,0]
    x_labels = ["katsavounidis", "xiaoxiao", "sa", "unitary", "random"]
    plt.errorbar(x_labels, interval_list[:,1], yerr=err, fmt='o')
    plt.show()
#
#    #rects1 = ax.bar(x + width, katsavounidis_mean, width, label='katsavounidis', yerr=katsavounidis_stddev)
#    #rects2 = ax.bar(x + 2 * width, xiaoxiao_mean, width, label='xiaoxiao', yerr=xiaoxiao_stddev)
#    #rects3 = ax.bar(x + 3 * width, sa_mean, width, label='sa', yerr=sa_stddev)
#    #rects4 = ax.bar(x + 4 * width, unitary_until_num_of_elements_mean, width, label='unitary', yerr=unitary_until_num_of_elements_stddev)
#    #rects5 = ax.bar(x + 5 * width, random_from_samples_mean, width, label='random', yerr=random_from_samples_stddev)
#
#    plt.boxplot(interval_list)
#
#    ax.set_ylabel('Minimal distortion')
#    ax.set_xlabel('Seed samples from trials')
#    ax.set_title('Minimal distortion by initial alphabet method - Nt = 16, k = 8000, var = 1.0')
#    ax.set_xticks(x)
#    #ax.set_xticklabels(labels)
#    ax.legend()
#
#    plt.show()
#
#    #print (sorted(random_from_samples_results))
#
#    ##norm_values_array_l1 = np.array(sorted(norm_values_l1, key=lambda k: k['norm'], reverse=True))
#    #norm_values_array_l1 = np.array([v['norm'] for v in norm_values_l1])
#    #norm_values_array_l1 = norm_values_array_l1/np.sqrt((np.sum(np.power(norm_values_array_l1, 2))))
#    #plt.plot(norm_values_array_l1, 'r*', label='variance = 0.1')
#
#    ##norm_values_array_l2 = np.array(sorted(norm_values_l2, key=lambda k: k['norm'], reverse=True))
#    #norm_values_array_l2 = np.array([v['norm'] for v in norm_values_l2])
#    #norm_values_array_l2 = norm_values_array_l2/np.sqrt((np.sum(np.power(norm_values_array_l2, 2))))
#    #plt.plot(xiaoxiao_v, '-', label='xiaoxiao_v')
#
#
#
#    #plt.savefig('results_graph1.png')
