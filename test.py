import matplotlib.pyplot as plt
import numpy as np

def plotEnergy(sumPSDs):
    #create labels
    #define threshold
    threshold = 0.8* np.max(sumPSDs)
    label = []
    for i in range(len(label)):
        name = 'Peak ' + str(i+1)
        label.append(name)
    plt.figure(8)
    plt.clf()
    index = np.arange(len(label))
    print(label)
    print(index.shape, sumPSDs.shape)
    # plt.bar(index, sumPSDs)

    # plt.plot(index, np.repeat(threshold, index.shape), color = 'r', linestyle = ':')
    # plt.ylabel('Energey of peaks (sum of psd)')
    # plt.xticks(index, label,rotation=30)
    # plt.title('Energy of each peak')



sumPSDs = np.array([0.1, 0.2, 0.4, 0.6])
plotEnergy(sumPSDs)

plt.show()
