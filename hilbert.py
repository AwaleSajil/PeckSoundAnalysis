import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert, chirp


duration = 1.0
fs = 400.0
samples = int(fs*duration)
t = np.arange(samples) / fs


signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1 + 0.5 * np.sin(2.0*np.pi*3.0*t) )
# non zero mean
signal += 12

signal = signal - np.average(signal)

analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)

plt.plot(t, signal)
plt.plot(t, amplitude_envelope)

plt.show()

# print(t)