import numpy as np

def gradient_descent(gradient, start, learn_rate, n_iter=50, tolerance=1e-06):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector

gradient = lambda v: 2 * v
start = 10.0
learn_rate = 0.8


result = gradient_descent(gradient, start, learn_rate, n_iter=50, tolerance=1e-06)
print (result)
