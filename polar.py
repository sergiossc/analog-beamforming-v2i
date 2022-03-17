import numpy as np
import matplotlib.pyplot as plt
from utils import *

def plot_unitary_codebook(codebook, filename):
    nrows, ncols = codebook.shape
    print (nrows, ncols)
    fig, axes = plt.subplots(nrows, ncols, subplot_kw=dict(polar=True))
    for col in range(ncols):
        for row in range(nrows):
            a = np.angle(codebook[row,col])
            r = np.abs(codebook[row,col])
            if nrows == 1:
                axes[col].plot(0, 1, 'wo')
                axes[col].plot(a, r, 'ro')
            else:
                axes[row, col].plot(0, 1, 'wo')
                axes[row, col].plot(a, r, 'ro')
    #plt.show()
    plt.savefig(filename)

def plot_codebook(codebook, filename):
    ncodewords, nrows, ncols = codebook.shape
    print (ncodewords, nrows, ncols)
    fig, axes = plt.subplots(ncodewords, ncols, subplot_kw=dict(polar=True))
    for col in range(ncols):
        for cw in range(ncodewords):
            a = np.angle(codebook[cw, 0, col])
            r = np.abs(codebook[cw, 0, col])
            axes[cw, col].plot(0, 1, 'wo')
            axes[cw, col].plot(a, r, 'ro')
    #plt.show()
    plt.savefig(filename)

def plot_samples(samples):
    nsamples, nrows, ncols = samples.shape
    fig, axes = plt.subplots(nrows, ncols, subplot_kw=dict(polar=True))

    for n in range(nsamples):
        for col in range(ncols):
            a = np.angle(samples[n, 0, col])
            r = np.abs(samples[n, 0, col])
            axes[col].plot(a, r, 'o')
    plt.show()

codebook = gen_dftcodebook(4)

samples = gen_samples(codebook, 8000, 0.1)

cb_avg = complex_average(samples)
#plot_codebook(cb_avg, 'cb_avg.png')
plot_unitary_codebook(cb_avg, 'initial_codebok.png')

cw0_shape = np.shape(cb_avg)
perturbation_variance = 0.1
perturbation_vector = np.sqrt(perturbation_variance/2) * (np.random.randn(cw0_shape[0], cw0_shape[1]) + 1j * np.random.randn(cw0_shape[0], cw0_shape[1]))

codebook2 = duplicate_codebook(cb_avg, perturbation_vector)

print (codebook2.shape)
plot_codebook(codebook2, 'cb2_avg.png')

codebook3 = duplicate_codebook(codebook2, perturbation_vector)
print (codebook3.shape)
plot_codebook(codebook3, 'cb3_avg.png')
#plot_codebook(codebook)
#plot_samples(samples)
#plot_codebook(cb_avg, "my_fig.png")
#plot_codebook(codebook[1:3], "my_fig.png")
