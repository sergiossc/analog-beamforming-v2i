import uuid
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from utils import *
import sys

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
    result_pathfile = sys.argv[1]


    with open(result_pathfile) as result:
        data = result.read()
        d = json.loads(data)

    distortion_by_round = d['mean_distortion_by_round'] 
    #distortion_by_round = dict2matrix(distortion_by_round)
    #print (distortion_by_round.shape)

    fig, ax = plt.subplots()
    for r, mean_distortion in distortion_by_round.items():
        mean_distortion = dict2matrix(mean_distortion)
        ax.plot(mean_distortion)
    plt.ylabel('distortion')
    plt.xlabel('# iterations')
    plt.show()

