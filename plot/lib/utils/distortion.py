import numpy as np

#def complex_squared_error(w, est_w):
def mean_squared_error(cw1, cw2):
    cw = cw1 - cw2
    squared_error = np.power(np.abs(cw), 2)
    
    return np.mean(squared_error) 
    #w_imag = w.imag
    #w_real = w.real
    #est_w_imag = est_w.imag
    #est_w_real = est_w.real
    #imag = w_imag - est_w_imag
    #real = w_real - est_w_real
    #squared_error = np.power(real,2) + np.power(imag,2) 
    #return np.sum(squared_error)

def distortion_mean(samples):
    """
    Input: A set of samples(semples) and the size of elements transmiters(num_tx) and receivers(num_rx).
    Output: A complex number as centroid of set samples. It consists of mean of real values plus mean of imaginary values.
    """
    samples_shape = samples.shape
    num_of_samples = samples_shape[0]
    axis0_of_sample = samples_shape[1]
    axis1_of_sample = samples_shape[2]

    sum_cw_imag = np.zeros((axis0_of_sample, axis1_of_sample))
    sum_cw_real = np.zeros((axis0_of_sample, axis1_of_sample))
   
    for sample in samples:
        sample_imag = sample.imag
        sample_real = sample.real
        sum_cw_imag += sample_imag
        sum_cw_real += sample_real
    mean_cw_imag = sum_cw_imag/num_of_samples
    mean_cw_real = sum_cw_real/num_of_samples
    mean_cw = mean_cw_real + mean_cw_imag * (1j)
    return mean_cw

def complex_distortion(sample, cw):
    """
    Input: complex matrix values sample and cw who has the same shape.
    Output: Return the distortion metric under each value of matrix. In this case is the Squared Error: (sample.img - cw.img)^2 + (sample.real - cw.real)^2.
    """
    #sample_imag = sample.imag
    #sample_real = sample.real
    #cw_imag = cw.imag
    #cw_real = cw.real
    #imag = sample_imag - cw_imag
    #real = sample_real - cw_real
    #complex_squared_error = np.power(imag,2) + np.power(real,2) 
    #distortion = complex_squared_error # distortion is a float, and positive
    #return np.sum(distortion)
    return mean_squared_error(sample, cw)

def dev_samples(samples):
    """
    Input: A set of samples(semples) and the size of elements transmiters(num_tx) and receivers(num_rx).
    Output: A complex number as centroid of set samples. It consists of mean of real values plus mean of imaginary values.
    """
    mean_cw = distortion_mean(samples)

    mean_cw_imag = mean_cw.imag
    mean_cw_real = mean_cw.real

    var_imag = np.var(mean_cw_imag)
    var_real = np.var(mean_cw_real)

    dev_imag = np.sqrt(var_imag)    
    dev_real = np.sqrt(var_real)    

    dev_cw = dev_real + dev_imag * (1j)
    return dev_cw


