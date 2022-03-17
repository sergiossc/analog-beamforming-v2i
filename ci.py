#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@author: sergiossc@gmail.com
"""
import numpy as np
import scipy.stats as st
from utils import conf_interval
import matplotlib.pyplot as plt

def conf_intervali_orig(x, conf_level):

    alpha = 1 - conf_level/100
    sample_size = len(x)
    sample_dist = np.sqrt(np.sum((x - np.mean(x)) ** 2)/(sample_size-1))

    if (sample_dist == 0.0 and sample_size < 30):
        print ('sample size too small for normal dist')
        return 0

    ci_values = None
    sample_mean = np.mean(x)
    sem = sample_dist/np.sqrt(sample_size) # Standard Error of Mean 

    if (sample_dist == 1.0 or sample_size < 30):
	# using T-student distribution
        if sample_size < 30:
            print(f'Small sample size: {sample_size}. It should be used only when the population has a Normal distribution.');
        ci_values = st.t.interval(conf_level, df=len(x)-1, loc=sample_mean, scale=sem)
        print (f't-student: {ci_values}')
    else:
        # using normal distribution
        ci_values = st.norm.interval(conf_level, loc=sample_mean, scale=sem)
        print (f'normal: {ci_values}')
    return sample_mean, ci_values

#x = np.array([-13.7,  13.1,  -2.8,  -1.1,  -3. ,   5.6])
#x = np.array([1.5, 2.6, -1.8, 1.3, -0.5, 1.7, 2.4])
x = np.array([3.1, 4.2, 2.8, 5.1, 2.8, 4.4, 5.6, 3.9, 3.9, 2.7, 4.1, 3.6, 3.1, 4.5, 3.8, 2.9, 3.4, 3.3, 2.8, 4.5, 4.9, 5.3, 1.9, 3.7, 3.2, 4.1, 5.1, 3.2, 3.9, 4.8, 5.9, 4.2])

#conf_levels = [0.90, 0.95, 0.99]
conf_level = 0.90
#for conf_level in conf_levels:
sample_mean, ci_values = conf_interval(x, conf_level)
print (f'sample_mean: {sample_mean}')
print (f'ci_values: {ci_values}')

fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
##x_set = [np.array(x), np.array(x)]
##ci_values_set = [ci_values, ci_values]
#ax1.set(xticks=['B', 'C'])
##ax1.boxplot(x_set, conf_intervals=ci_values_set, notch=True, labels=("A","B"))
#ax1.boxplot(np.array(x), conf_intervals=[ci_values], notch=True)
#ax1.boxplot(np.array(x), notch=True)
##ax1.plot(np.arange(2), np.ones(2) * ci_values[0])

##ax1.plot(np.arange(2), np.ones(2) * ci_values[1])
yerr = ci_values[1] - ci_values[0]
ax1.errorbar(2, sample_mean, yerr=yerr/2, capsize=5, marker='o')

ax1.plot(np.arange(2), np.ones(2) * np.median(x), label='median')
ax1.plot(np.arange(2), np.ones(2) * np.mean(x), label='mean', marker='+')
ax1.plot(np.arange(2), np.ones(2) * np.max(x), label='max')
ax1.plot(np.arange(2), np.ones(2) * np.min(x), label='min')
ax1.plot(np.arange(2), np.ones(2) * ci_values[0], label='ci_low')
ax1.plot(np.arange(2), np.ones(2) * ci_values[1], label='ci_high')
plt.legend()
plt.show()
