import sys
import numpy as np
#import load_lib
#from lib.mimo import arrayconfig as ac
#from utils import PlotPattern2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #plot.PlotPattern2()
    #w = np.random.rand(ac.tx_array.size) * 1j + np.random.rand(ac.tx_array.size)
    #print (w)
    #plot.plot_pattern(ac.tx_array, w)
    #PlotPattern2(ac.tx_array)
    ax_pos_dict = {4: (0,0), 8: (0,1), 16: (1,0), 32: (1,1), 64:(2,0), 128:(2,1), 256:(3,0), 512:(3,1)}
    
    fig, ax = plt.subplots(4,2, gridspec_kw = {'wspace':0.15, 'hspace':0.25})
    fig.supxlabel('XLAgg')
    fig.supylabel('YLAgg')
    print (np.shape(ax))
    x4 = np.random.randn(50)
    ax[ax_pos_dict[512]].plot(x4, label='x4')
    ax[ax_pos_dict[512]].legend(loc='best', fontsize='11')
    plt.show()
