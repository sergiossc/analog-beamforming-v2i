"""
   This file contains some general config values applied to training, validation, and test processes.
"""
# GENERAL PARAMETERS
kfold_number = 5
data_to_use = 1.0
#dataset2use = 'textfiles'
dataset_type = 'hdf5'
create_validation_set = True
get_samples_by_user = False # Set True if you want to get npy sample files by RX. Interesting for non-mobile transceivers
angularize = False # Array domain channels became angular domain

# hdf5
#prefix_episodefiles = '/home/snow/github/land/dataset/s000/Raymobtime_Dataset/Raymobtime_s000/ray_tracing_data_s000_carrier60GHz'
#dataset_name = 's000'

prefix_episodefiles = '/home/snow/github/land/dataset/s002/ray_tracing_data_s002_carrier60GHz'
dataset_name = 's002'

#dataset_name = 's004'
#prefix_episodefiles = '/home/snow/github/land/dataset/s004/ray_tracing_data_s004_carrier60GHz'

#prefix_episodefiles = '/home/snow/github/land/dataset/s006/ray_tracing_data_s006_carrier60GHz'
#dataset_name = 's006'

#prefix_episodefiles = '/home/snow/github/land/dataset/s007/ray_tracing_data_s007_carrie60GHz'
#dataset_name = 's007'

#prefix_episodefiles = '/home/snow/github/land/dataset/s008/ray_tracing_data_s008_carrier60GHz'
#dataset_name = 's008'

#prefix_episodefiles = '/home/snow/github/land/dataset/s009/ray_tracing_data_s009_carrier60GHz'
#dataset_name = 's009'
