import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy import stats
import numpy as np
import pandas as pd

x = np.array([242, 228, 217, 209, 253, 239, 266, 242, 251, 240, 223, 219, 246, 260, 258, 225, 234, 230, 249, 245, 254, 243, 235, 231, 257])

df = pd.DataFrame(x)

print (df)

df.hist(bins=6)
plt.show()


#rng = default_rng()
#x = [1, 4, 2, 1, 3, 1]
#res = stats.cumfreq(x, numbins=4, defaultreallimits=(1.5, 5))
#res.cumcount
# array([ 1.,  2.,  3.,  3.])
#res.extrapoints
# 3

# Create a normal distribution with 1000 random values

#samples = stats.norm.rvs(size=1000, random_state=rng)

# Calculate cumulative frequencies

#res = stats.cumfreq(x, numbins=)
#
## Calculate space of values for x
#
#x = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size,
#                                 res.cumcount.size)
#
## Plot histogram and cumulative histogram
#
#fig = plt.figure(figsize=(10, 4))
#ax1 = fig.add_subplot(1, 2, 1)
#ax2 = fig.add_subplot(1, 2, 2)
#ax1.hist(x, bins=7)
#ax1.set_title('Histogram')
#ax2.bar(x, res.cumcount, width=res.binsize)
#ax2.set_title('Cumulative histogram')
#ax2.set_xlim([x.min(), x.max()])
#plt.grid()
#plt.show()
