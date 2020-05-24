import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy import signal
import time
import datetime


RATE = None
data = None


RATE, data = read("Feedday.wav")

data = (data + (2**16)/2)/300




xx = np.linspace(0, 0.24*1000, 0.24*RATE)/10

# print(xx)
#chop data
start = int(0.06*RATE)
end = int(0.18*RATE)
data = data[start:end]
xx = xx[start:end]





sumofPecks = np.sum(data)
totalFeedPerDay = (sumofPecks*0.025)/1000
print(totalFeedPerDay)


data = data*0.025


plt.figure("2")
plt.plot(xx, data, color = '#1a9988')

locs, labels = plt.xticks()           # Get locations and labels
plt.xticks([6, 8, 10, 12, 14, 16, 18], ["6:00", "8:00", "10:00", "12:00", "14:00", "16:00", "18:00"])  # Set locations and labels

# plt.plot(fftVal[0][peakIndex], fftVal[1][peakIndex])
plt.title('Feeding consumption vs Time Graph (#pecks to gram)')
plt.xlabel('Time of the Day')
plt.ylabel('Feeding consumption (gm)')
plt.tight_layout()

plt.savefig('fig10.png', transparent=True)


plt.show()




