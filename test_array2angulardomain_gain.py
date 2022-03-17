import numpy as np
from utils import array_to_angular_domain, get_beamforming_gain, gen_channel
from numpy.linalg import norm
import matplotlib.pyplot as plt

nr = 4
nt = 4


num_samples = 100

gain_ch_v = []
gain_ch_angular_v = []
for i in range(num_samples):
    ch = gen_channel(nr, nt, 1)
    ch = np.sqrt(nr * nt) * ch/norm(ch)
    ch_angular = array_to_angular_domain(ch)
    gain_ch = get_beamforming_gain(ch)
    gain_ch_angular = get_beamforming_gain(ch_angular)
    print (gain_ch)
    print (type(gain_ch))
    gain_ch_v.append(gain_ch)
    gain_ch_angular_v.append(gain_ch_angular)

#plt.ticklabel_format(useOffset=False)
plt.plot(gain_ch_v, label='gain_ch_v')
plt.plot(gain_ch_angular_v, label='gain_ch_angular_v')
plt.legend()
plt.show()
