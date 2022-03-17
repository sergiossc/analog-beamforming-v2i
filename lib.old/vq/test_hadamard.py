from utils import hadamard_transform
import numpy as np

x = np.arange(8).reshape(1,1,8)
x = x + 1j * x

print (x)

x_hadamard = hadamard_transform(x)
print (x_hadamard)

x_hadamard_inv = hadamard_transform(x_hadamard, True)
print (x_hadamard_inv)
