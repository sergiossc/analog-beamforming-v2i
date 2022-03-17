import pandas as pd
import numpy as np
import os
import uuid
import scipy.io as io

def check_textfiles(prefix, scenes_dir, posfix1, posfix2):
    pathfiles = {}
    for scene_file in scenes_dir:
        pathfile1 = prefix + str(scene_file) + posfix1
        pathfile2 = prefix + str(scene_file) + posfix2
        posfix1_status =False
        posfix2_status = False
        try:
            current_file1 = open(pathfile1)
            posfix1_status = True
            #print("Sucess.")
        except IOError:
            print("File not accessible: ", pathfile1)
        finally:
            current_file1.close()

        try:
            current_file2 = open(pathfile2)
            posfix2_status = True
            #print("Sucess.")
        except IOError:
            print("File not accessible: ", pathfile2)
        finally:
            current_file2.close()

        if (posfix1_status == True and posfix1_status == True):
            pathfile_id = uuid.uuid4()
            pathfiles[pathfile_id] = prefix + str(scene_file)
 
    return pathfiles

def check_hdf5files(prefix, episodefiles):
    pathfiles = {}
    for ep_file in episodefiles:
        pathfile = prefix + str('/') + str(ep_file)
        ep_file_status = False
        try:
            current_file = open(pathfile)
            ep_file_status = True
            print(f'Cheking {pathfile}... Ok!')
        except IOError:
            print("File not accessible: ", pathfile)
        finally:
            current_file.close()

        if ep_file_status:
            ep_file_id = uuid.uuid4()
            pathfiles[ep_file_id] = pathfile
 
    return pathfiles



#def preprocessing_textfiles(kfold_number, prefix_pathfiles, paths_posfix, pg_posfix, data_to_use):
#    #print ('textfiles ----- >>>>> ')
#
#    #Get some scenes from all scenes (percent of data to use)
#    scenes_dir = os.listdir(prefix_pathfiles)
#    scenes_dir = np.random.choice(scenes_dir, int(len(scenes_dir) * data_to_use), replace=False)
#
#    # check is scenes of paths are OK
#    pathfiles = check_textfiles(prefix_pathfiles, scenes_dir, paths_posfix, pg_posfix)
#    
#    # check if pathgain files are OK
#    #pg_pathfiles = check_file(prefix_pathfiles, scenes_dir, pg_posfix)
#
#    num_of_files = len(pathfiles)
#    #print('num_of_files: ', num_of_files)
#    kfold_slot_length = int(num_of_files/kfold_number)
#
#    pathfiles_id_list = [pathfiles_id for pathfiles_id in pathfiles.keys()]
#    #print(pathfiles_id_list)
#    val_pathfile_ids = np.random.choice(pathfiles_id_list, kfold_slot_length, replace=False)
#    validation_set = {}
#    for val_pathfile_id in val_pathfile_ids:
#        validation_set[val_pathfile_id] = pathfiles.pop(val_pathfile_id)
#    #print('validation_set.num_of_files: ', len(validation_set))
#
#    pathfiles_id_list = [pathfiles_id for pathfiles_id in pathfiles.keys()]
#    test_pathfile_ids = np.random.choice(pathfiles_id_list, kfold_slot_length, replace=False)
#    testing_set = {}
#    for test_pathfile_id in test_pathfile_ids:
#        testing_set[test_pathfile_id] = pathfiles.pop(test_pathfile_id)
#    #print('testingn_set.num_of_files: ', len(testing_set))
#
#    training_set = pathfiles
#    #print('training_set.num_of_files: ', len(training_set))
#
#    df_training = pd.DataFrame(data=training_set, index=[0])
#    df_training.to_csv('training_set.csv')
#        
#    df_validation = pd.DataFrame(data=validation_set, index=[0])
#    df_validation.to_csv('validation_set.csv')
#
#    df_testing = pd.DataFrame(data=testing_set, index=[0])
#    df_testing.to_csv('testing_set.csv')
#
#
##prefix_pathfiles = '/home/snow/github/land/dataset/textfiles/'
##paths_posfix = '/model.paths.t001_01.r002.p2m'
##pg_posfix = '/model.pg.t001_01.r002.p2m'
##kfold_number = 5
##data_to_use = 0.08
###data_to_use = 0.0045 # How much from dataset to use. '1' is 100% of data
##preprocessing(kfold_number, prefix_pathfiles, paths_posfix, pg_posfix, data_to_use)
#
#
#def preprocessing_hdf5(kfold_number, prefix_episodefiles, data_to_use, create_validation_set=True):
#    #print ('Preprocessing hdf5 files... ')
#    #print ('Create Validation episode group is ', create_validation_set)
#
#    #Get some scenes from all scenes (percent of data to use)
#    episode_files = os.listdir(prefix_episodefiles)
#    episode_files = np.random.choice(episode_files, int(len(episode_files) * data_to_use), replace=False)
#
#    # check if the episodes pathfiles are OK
#    pathfiles = check_hdf5files(prefix_episodefiles, episode_files)
#
#    num_of_files = len(pathfiles)
#    #print('num_of_files: ', num_of_files)
#    kfold_slot_length = int(num_of_files/kfold_number)
#
#    
#    if create_validation_set:
#        pathfiles_id_list = [pathfiles_id for pathfiles_id in pathfiles.keys()]
#        val_pathfile_ids = np.random.choice(pathfiles_id_list, kfold_slot_length, replace=False)
#        validation_set = {}
#        for val_pathfile_id in val_pathfile_ids:
#            validation_set[val_pathfile_id] = pathfiles.pop(val_pathfile_id)
#        #print('validation_set.num_of_files: ', len(validation_set))
#    
#        pathfiles_id_list = [pathfiles_id for pathfiles_id in pathfiles.keys()]
#        test_pathfile_ids = np.random.choice(pathfiles_id_list, kfold_slot_length, replace=False)
#        testing_set = {}
#        for test_pathfile_id in test_pathfile_ids:
#            testing_set[test_pathfile_id] = pathfiles.pop(test_pathfile_id)
#        #print('testingn_set.num_of_files: ', len(testing_set))
#    
#        training_set = pathfiles
#        #print('training_set.num_of_files: ', len(training_set))
#    
#        df_training = pd.DataFrame(data=training_set, index=[0])
#        df_training.to_csv('hdf5_training_set.csv')
#            
#        df_validation = pd.DataFrame(data=validation_set, index=[0])
#        df_validation.to_csv('hdf5_validation_set.csv')
#    
#        df_testing = pd.DataFrame(data=testing_set, index=[0])
#        df_testing.to_csv('hdf5_testing_set.csv')
#    else:
#        pathfiles_id_list = [pathfiles_id for pathfiles_id in pathfiles.keys()]
#        test_pathfile_ids = np.random.choice(pathfiles_id_list, kfold_slot_length, replace=False)
#        testing_set = {}
#        for test_pathfile_id in test_pathfile_ids:
#            testing_set[test_pathfile_id] = pathfiles.pop(test_pathfile_id)
#        #print('testing_set.num_of_files: ', len(testing_set))
#    
#        training_set = pathfiles
#        #print('training_set.num_of_files: ', len(training_set))
#    
#        df_training = pd.DataFrame(data=training_set, index=[0])
#        df_training.to_csv('hdf5_training_set.csv')
#            
#        df_testing = pd.DataFrame(data=testing_set, index=[0])
#        df_testing.to_csv('hdf5_testing_set.csv')
#        
#    
#    
