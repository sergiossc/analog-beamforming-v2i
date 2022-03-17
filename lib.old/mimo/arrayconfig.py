"""
   This file contains some general config values applied to training, validation, and test processes.
"""
#import sys
#sys.path.append(r'./lib/mimo/')
#sys.path.append(r'./lib/mimo/utils')
from scipy.constants import c
from mimo.phased import PartitionedArray

# ANTENNA CONFIG
   
num_rx = 4 # 
num_tx = 64 # 

num_stream = 1 # 

fc = 60*(10**9) # 
wave_length = c/fc # 

element_spacing = wave_length/2 # 

tx_array = PartitionedArray(num_tx, element_spacing, wave_length, "UPA") # 
rx_array = PartitionedArray(num_rx, element_spacing, wave_length, "UPA") # 
