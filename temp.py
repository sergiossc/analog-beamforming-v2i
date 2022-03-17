import load_lib
import numpy as np

import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import pandas as pd
from utils import gen_hermitian_matrix
 
if __name__ == '__main__':

    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True) 

    for ax in axes.flat:
    
        n = 4
        m = 4

        #h = np.random.rand(n, m)
        h = np.abs(gen_hermitian_matrix(n))

        dtheta = np.arange(n)
        dphi = np.arange(n)

        df = pd.DataFrame(h, index=dtheta, columns=dphi)
        sns.heatmap(df, ax=ax, cmap='PiYG')

    plt.show()
