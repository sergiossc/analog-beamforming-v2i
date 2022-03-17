import numpy as np
from matplotlib import pyplot as plt
#from scipy.io.wavfile import write
from scipy.fft import fft, fftfreq


gridsize = (2, 2)
fig = plt.figure(figsize=(7,7))

ax1 = plt.subplot2grid(gridsize, (0, 0))
ax2 = plt.subplot2grid(gridsize, (0, 1))
ax3 = plt.subplot2grid(gridsize, (1, 1))


SAMPLE_RATE = 44100 # Hz
DURATION = 5 # s

def gen_sinewave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

_, nice_tone = gen_sinewave(400, SAMPLE_RATE, DURATION)
_, noise_tone = gen_sinewave(4000, SAMPLE_RATE, DURATION)

noise_tone = noise_tone * 0.3

mixed_tone = nice_tone + noise_tone
mixed_tone = mixed_tone/np.max(mixed_tone)
print (np.sum(mixed_tone))

#plt.plot(mixed_tone[0:1000])
#plt.show()

# Remember SAMPLE_RATE = 44100 Hz is our playback rate
#write("mysinewave.wav", SAMPLE_RATE, mixed_tone)

N = SAMPLE_RATE * DURATION

yf = fft(mixed_tone)
xf = fftfreq(N, 1/SAMPLE_RATE)
ax1.plot(xf, np.abs(yf))

# Filtering the Signal
# The maximum frequency is half the sample rate
points_per_freq = len(xf) / (SAMPLE_RATE / 2)
# Our target frequency is 4000 Hz
target_idx = int(points_per_freq * 4000)
yf[target_idx - 1 : target_idx + 2] = 0

ax2.plot(xf, np.abs(yf))
ax2.set_title('filtered')

plt.show()

