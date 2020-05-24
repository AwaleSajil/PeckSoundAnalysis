import numpy as np


bul = np.array([0,0,1,1,1,0,0,0,0,1,1,0,0,0,1,1,0,0])
d = np.array  ([1,3,2,4,3,5,4,6,2,8,6,7,5,9,5,9,0,7])

def splitPeaks(data, bulen):
    peaks = []

    for i in range(data.shape[0] - 1):
        if bulen[i] ==0 and bulen[i+1] == 1:
            #raising edge
            tmp = np.array([])
        elif bulen[i] == 1 and bulen[i+1] == 1:
            tmp  = np.concatenate((tmp, np.array([data[i]])))
        elif bulen[i] == 1 and bulen[i+1] == 0:
            #falling edge
            tmp  = np.concatenate((tmp, np.array([data[i]])))
            print(tmp)
            # peaks = np.concatenate((peaks, tmp), axis = 0)
            peaks.append(tmp)
            tmp = np.array([])
        elif data[i] == 0 and data[i+1] ==0:
            pass
    return peaks


x = splitPeaks(d, bul)
print(x)