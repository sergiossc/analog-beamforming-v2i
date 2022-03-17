import uuid
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from utils import *
import sys


if __name__ == '__main__':

    #profile_pathfile = 'profile.json' 
    result_pathfile = sys.argv[1]

    with open(result_pathfile) as result:
        data = result.read()
        d = json.loads(data)

    initial_alphabet_opt = d['initial_alphabet_opt']
    distortion_by_round = d['mean_distortion_by_round'] 

    fig, ax = plt.subplots()
    ax.ticklabel_format(useOffset=False)
    for r, mean_distortion in distortion_by_round.items():
        mean_distortion = dict2matrix(mean_distortion)
        ax.plot(mean_distortion)
    plt.ylabel('mse')
    plt.xlabel('# iterations')
    plt.title(f'initial alphabet opt: {initial_alphabet_opt}')
    plt.show()
