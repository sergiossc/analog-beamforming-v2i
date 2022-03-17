import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print ('test matplotlib errorbar')
    np.random.seed(5678)
    x = np.random.randn(10000,10)
    print (np.shape(x))
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    yerr = x_std
    #yerr = np.vstack((x_mean - x_std, x_mean + x_std))
    plt.errorbar(np.arange(10), x_mean, yerr=yerr, marker='o')
    plt.show()
    
