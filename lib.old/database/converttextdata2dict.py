import pandas as pd
import numpy as np
import os
import uuid
import scipy.io as io
import re
import time
import sqlite3
from tqdm import tqdm

def readtransceiversfromtextfiles(pathfile):
    #pathfile = 'model.paths.t001_01.r002.p2m'
    file_content = {}
    transceivers = {}
    regex = '^[0-9]'
    try:
        my_file = open(pathfile)
        count_line = 0
        for l in my_file:
            if (re.search(regex, l)):
                line = l.split()
                file_content[count_line] = line
                count_line += 1
    except IOError:
        print('Cannot open')
    finally:
        my_file.close()

    # Each scene has only a single Tx transceiver, and his position don't change
    transceiver_record = {}
    transceiver_record['label'] = '1'
    transceiver_record['type'] = 'tx'
    transceiver_record['posx'] = 746.0
    transceiver_record['posy'] = 560.0
    transceiver_record['posz'] = 42.523550671279
    transceiver_record['distance'] = 0
    transceiver_record['pathgain'] = 0
    
    transceiver_id = uuid.uuid4()
    transceivers[transceiver_id] = transceiver_record
 
    for line_number, line in file_content.items():
        transceiver_record = {}
        if len(line) == 6:
            #print (line)
            transceiver_record['label'] = line[0]
            transceiver_record['type'] = 'rx'
            transceiver_record['posx'] = line[1]
            transceiver_record['posy'] = line[2]
            transceiver_record['posz'] = line[3]
            transceiver_record['distance'] = line[4]
            transceiver_record['pathgain'] = line[5]

            transceiver_id = uuid.uuid4()
            transceivers[transceiver_id] = transceiver_record
 
    return transceivers


def readpathsfromtextfiles(pathfile):
    """
    Input: pathfile model.paths.*.p2m 
    Output: get rays information from scene: AoA, DoD, received_power, number of receivers, transmiters
    """
    #pathfile = 'model.paths.t001_01.r002.p2m'
    file_content = {}
    regex = '^[0-9]'
    try:
        my_file = open(pathfile)
        count_line = 0
        for l in my_file:
            if (re.search(regex, l)):
                line = l.split()
                file_content[count_line] = line
                count_line += 1
    except IOError:
        print('Cannot open')
    finally:
        my_file.close()
    paths = {}
    rx_label = ''
    for line_number, line in file_content.items():
        path_record = {}
    
        if len(line) == 2:
            rx_label  = line[0]
        if len(line) == 8:
            path_record['label'] = line[0]
            path_record['tx_label'] = '1'
            path_record['rx_label'] = rx_label
            path_record['received_power'] = line[2]
            path_record['arrival_theta'] = line[4]
            path_record['arrival_phi'] = line[5]
            path_record['departure_theta'] = line[6] #np.random.choice([60,170]) #line[6]
            path_record['departure_phi'] = line[7]
            path_record['channel'] = '1' + str(rx_label)

            path_id = uuid.uuid4()
            paths[path_id] = path_record
    return paths
