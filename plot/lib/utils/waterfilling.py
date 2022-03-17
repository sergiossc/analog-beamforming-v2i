"""
Water-filling power allocation
"""
import numpy as np
import uuid

class channel():
    def __init__(self, eigenvalue, noise_var, power):
        self.id = uuid.uuid4()
        self.eigenvalue = eigenvalue
        #self.eigenvalue = eigenvalue * 1.0e30
        self.noise_var = noise_var
        self.power = power
        self.enable = False


def wf(eigen_values, total_pwr, noise_var):
    """
       Power allocation using water-filling.:
         Parameters:
          channels: array-like of channels with information of eigenvalues and noise variance of each channel.
          total_pwr: float total power of base station transmiter.
         Return:
          Void: set power property of each channel in channels array.
    """
    channels = []
    for eg in eigen_values:
        ch = channel(eg, noise_var, -100)
        #ch = channel(eg, np.sqrt(1/2)*np.random.randn(), -100)
        channels.append(ch)

    wf_list = []

    for ch in channels:
        wf_rec = {"ch": ch, "noise_lambda": 0.0}
        #print("channel.lambda: ", ch.eigenvalue)
        wf_list.append(wf_rec)

    for wf in wf_list:
        ch = wf["ch"]
        wf["noise_lambda"] = ch.noise_var/ch.eigenvalue


    previous_cut_factor = -100
    cut_factor = 0

    while True:
        wf_list_disable =  [ w for w in wf_list if w["ch"].enable == False]
        previous_cut_factor = cut_factor
        cut_factor = (  total_pwr + np.sum( [ wf["noise_lambda"] for wf in wf_list if wf["ch"].enable == False] )   ) / len (  wf_list_disable  ) 

        for wf in wf_list_disable:
            ch = wf["ch"]
            if wf["noise_lambda"] >= cut_factor:
                ch.enable = True
                ch.power = 0.0

        if cut_factor == previous_cut_factor:
            for wf in wf_list_disable:
                ch = wf["ch"]
                ch.power = cut_factor - wf["noise_lambda"]
                ch.enable = True
            break
    d = np.array([ch.power for ch in channels])
    d = np.sort(d)
    d = d[::-1]
    return d, cut_factor
