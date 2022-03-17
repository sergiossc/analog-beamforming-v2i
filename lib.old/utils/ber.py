import numpy as np

def ber_calc(a, b):
    num_ber = np.sum(np.abs(a - b))  # error sum
    ber = np.mean(np.abs(a - b))
    return int(num_ber), ber


#a = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1])
#b = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])

#num_err, ber = ber_calc(a,b)

#print ("num_err: ", num_err)
#print ("ber: ", ber)
