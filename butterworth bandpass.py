from scipy.signal import butter, lfilter, freqz
import numpy as np
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


#signal
fs = 16000
T = 0.01
samples = T*fs
t = np.linspace(0, T, samples, endpoint=False)
x = np.sin(2*3.14*500*t) + np.sin(2*3.14*4000*t) + np.cos(2*3.14*7000*t)

lowcut = 1000
highcut = 5000
y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)


plt.plot(t,x)
plt.show()

plt.plot(t,y)
plt.show()
