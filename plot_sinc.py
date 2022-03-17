import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    srate = 1000
    time = np.linspace(0, 2, srate)
    points = len(time)
    signal = 2.5 * np.sin(2 * np.pi * 4 * time) + (1.5 * np.sin(2 * np.pi * 6.5 * time))

    plt.plot(time, signal)
    plt.show()