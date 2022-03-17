import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import optimize
import seaborn as sns
sns.set(style="whitegrid")

def plt_rv_dist(X, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1,3, figsize=(12,3))

    x_min_999, x_max_999 = X.interval(0.999)
    x999 = np.linspace(x_min_999, x_max_999, 1000)

    x_min_95, x_max_95 = X.interval(0.95)
    x95 = np.linspace(x_min_95, x_max_95, 1000)

    if hasattr(X.dist, "pdf"):
        axes[0].plot(x999, X.pdf(x999), label="PDF") 
        axes[0].fill_between(x95, X.pdf(x95), alpha=0.25)
    else:
        x999_int = np.unique(x999.astype(int))
        axes[0].bar(x999_int, X.pmf(x999_int), label="PMF")
    axes[1].plot(x999, X.cdf(x999), label="CDF")
    axes[1].plot(x999, X.sf(x999), label="SF")
    axes[2].plot(x999, X.ppf(x999), label="PPF")

    for ax in axes:
        ax.legend()

if __name__ == "__main__":
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    X = stats.norm()
    plt_rv_dist(X, axes=axes[0,:])
    axes[0, 0].set_ylabel("Normal dist.")

    X = stats.f(2, 50)
    plt_rv_dist(X, axes=axes[1,:])
    axes[1, 0].set_ylabel("F dist.")


    X = stats.poisson(5)
    plt_rv_dist(X, axes=axes[2,:])
    axes[2, 0].set_ylabel("Poisson dist.")

    plt.show()
    