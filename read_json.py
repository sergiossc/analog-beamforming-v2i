import uuid
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from utils import *
from lib.utils.plot import plot_devices

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

    device_pos_info_filename = 'device_pos_info.json' 
    with open(device_pos_info_filename) as device_pos_info:
        data = device_pos_info.read()
        d = json.loads(data)

    transceivers = []

    device0 = d['device0']
    dev0_label = d['dev0_label']
    dev0_posx = d['dev0_posx']
    dev0_posy = d['dev0_posy']
    dev0_posz = d['dev0_posz']
    transceiver1 = {'type': device0, 'label': dev0_label, 'posx': dev0_posx, 'posy': dev0_posy, 'posz': dev0_posz} 
    transceivers.append(transceiver1)


    device4 = d['device4']
    dev4_label = d['dev4_label']
    dev4_posx = d['dev4_posx']
    dev4_posy = d['dev4_posy']
    dev4_posz = d['dev4_posz']
    transceiver4 = {'type': device4, 'label': dev4_label, 'posx': dev4_posx, 'posy': dev4_posy, 'posz': dev4_posz} 
    transceivers.append(transceiver4)


    device5 = d['device5']
    dev5_label = d['dev5_label']
    dev5_posx = d['dev5_posx']
    dev5_posy = d['dev5_posy']
    dev5_posz = d['dev5_posz']
    transceiver5 = {'type': device5, 'label': dev5_label, 'posx': dev5_posx, 'posy': dev5_posy, 'posz': dev5_posz} 
    transceivers.append(transceiver5)

    for transceiver in transceivers:
        print (transceiver)
    plot_devices(transceivers)
