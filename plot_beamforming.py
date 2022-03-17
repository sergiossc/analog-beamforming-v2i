import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c


if __name__ == '__main__':

    nr = 10
    fc = 60 * (10 ** 9) # 60 GHz
    wavelenght = c/fc
    d = wavelenght/2 # d = 0.5
    rx_array = np.arange(nr)

    theta = np.linspace(-2*np.pi, 2*np.pi, 100)
    theta_user = np.pi/8

    for _ in range(len(theta)):
        er = np.exp(-1j * 2 * np.pi * d * (1/wavelenght) * np.cos(theta[_]))


    plt.plot(theta)
    plt.show()
