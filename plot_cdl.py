import load_lib
import os
import numpy as np
import sys

import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import pandas as pd
 

if __name__ == '__main__':

    npy_pathfile1 = sys.argv[1]
    npy_pathfile2 = sys.argv[2]

    paths1 = np.load(npy_pathfile1, allow_pickle=True)
    paths2 = np.load(npy_pathfile2, allow_pickle=True)

    paths1 = paths1[23]
    print (np.shape(paths1))
    paths2 = paths2[45]
    print (np.shape(paths2))

    plot_flag = True
    
    if plot_flag:
        gridsize = (2, 2)
        fig = plt.figure(figsize=(7,7))
        
        #ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
        ax1 = plt.subplot2grid(gridsize, (0, 0))
        ax2 = plt.subplot2grid(gridsize, (0, 1))
        ax3 = plt.subplot2grid(gridsize, (1, 0))
        ax4 = plt.subplot2grid(gridsize, (1, 1))
    
        
        df1 = pd.DataFrame(paths1, columns=['aoa theta', 'aoa phi', 'aod theta', 'aod phi', 'received power'])
        df1.plot.scatter(subplots=True, ax=ax1, sharex=False, sharey=False, title=['RX4'], xlabel='phi', ylabel='theta', grid=True, x=['aoa phi'], y=['aoa theta'])
        df1.plot.scatter(subplots=True, ax=ax2, sharex=False, sharey=False, title=['TX (RX4)'], xlabel='phi', ylabel='theta', grid=True, x=['aod phi'], y=['aod theta'])
    
        df2 = pd.DataFrame(paths2, columns=['aoa theta', 'aoa phi', 'aod theta', 'aod phi', 'received power'])
        df2.plot.scatter(subplots=True, ax=ax3, sharex=False, sharey=False, title=['RX5'], xlabel='phi', ylabel='theta', grid=True, x=['aoa phi'], y=['aoa theta'])
        df2.plot.scatter(subplots=True, ax=ax4, sharex=False, sharey=False, title=['TX (RX5)'], xlabel='phi', ylabel='theta', grid=True, x=['aod phi'], y=['aod theta'])
     
        #print (df)
        #plt.legend()
        plt.show()
    
        
