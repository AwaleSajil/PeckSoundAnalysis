import pyaudio
import numpy as np
import wave
import math
import scipy

import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert, butter, lfilter, freqz


def matchShape(a):
    global signal
    d = signal.shape[0] - a.shape[0]
    if d > 0:
        ex = np.zeros(d)
        a = np.concatenate((a, ex))
    return a

def movingAverage(values, window):
    weights = np.repeat(1.0, window)/window
    smas = np.convolve(values, weights, 'valid')
    return smas


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
            # print(tmp)
            # peaks = np.concatenate((peaks, tmp), axis = 0)
            peaks.append(tmp)
            tmp = np.array([])
        elif data[i] == 0 and data[i+1] ==0:
            pass
    return peaks



File='signal+.wav'
chunk = int(44100*1)
rate = 44100
spf = wave.open(File, 'rb')
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
#normalize the signal
signal = signal/(2**15)

#chop the signal for study in time axis
signal = signal[1:chunk]

#define time
t = (np.linspace(1, int(signal.shape[0]), num=int(signal.shape[0])))/rate

#correct signal mean to zero
signal = signal - np.average(signal)


#adaptivethreshold
adpThreshold = None
#moving average 
avg = None
#boolen
boolLoc = None


def bandpass_1():
    global signal
    #STEP 1 BandPass Filter
    plt.figure(1)
    plt.clf()
    plt.plot(t, signal,  color = '#1a9988',  linewidth=1)
    plt.title('Original Signal (recording)')
    plt.xlabel('time(sec)')
    plt.ylabel('amp')
    plt.ylim(-1, 1)
    plt.tight_layout()
    # plt.savefig('fig1.png', transparent=True)
    


    #perform bandpass filter
    lowcut = 2000
    highcut = 4000
    signal = butter_bandpass_filter(signal, lowcut, highcut, rate, order=6)

    plt.figure(2)
    plt.clf()
    plt.plot(t, signal, color = '#1a9988',  linewidth=1)
    plt.title('Bandpassed Signal')
    plt.xlabel('time(sec)')
    plt.ylabel('amp')
    plt.ylim(-1, 1)
    plt.tight_layout()
    # plt.savefig('fig2.png', transparent=True)



def hilbert_2():
    global signal, adpThreshold,avg
    #perform hilbert transform
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    #plot it
    plt.figure(3)
    plt.clf()
    plt.plot(t, signal, color = '#1a9988', label = 'signal')
    plt.plot(t, amplitude_envelope, color = '#eb5600',  linewidth=1, label = 'envelope')
    plt.title('Hilbert Transform')
    plt.xlabel('time(sec)')
    plt.ylabel('amp')
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.legend()
    # plt.savefig('fig3.png', transparent=True)

    #perform moving average
    avg = movingAverage(amplitude_envelope, 1200)
    #match the shape of avg with signal or t
    avg = matchShape(avg)
    
    #find the threshold
    adpThreshold = 0.8*np.max(avg)

    #plot it
    plt.figure(4)
    plt.clf()
    plt.plot(t, signal, color = '#1a9988', label = 'signal')
    plt.plot(t[0:avg.shape[0]], avg, color = '#eb5600',  linewidth=1, label ='smooth envelope')
    plt.plot(t, np.repeat(adpThreshold, t.shape), linewidth=1, linestyle=':', color= 'b', label = 'Adaptive Threshold')
    plt.title('Moving average of envelope from hilbert transform')
    plt.xlabel('time(sec)')
    plt.ylabel('amp')
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.legend()
    # plt.savefig('fig4.png', transparent=True)

    # plt.plot(t[0:avg.shape[0]], avg)

    # plt.plot((np.linspace(1, int(signal.shape[0]), num=int(signal.shape[0])))/rate, signal)
    # plt.show()


def filter_3():
    global signal, adpThreshold, boolLoc

    #perform the filter
    boolLoc = (avg) >= adpThreshold
    signal = signal*boolLoc

    #plot it
    plt.figure(5)
    plt.clf()
    plt.plot(t, signal, color = '#1a9988', )
    #plot boolLoc
    plt.plot(t, boolLoc, color = '#ff0000', linestyle = ':', linewidth = 0.5)
    plt.title('Trimmed signal based on adaptive_threshold = 0.8*maxOfEnvelope')
    plt.xlabel('time(sec)')
    plt.ylabel('amp')
    plt.ylim(-1, 1)
    plt.tight_layout()
    # plt.savefig('demo1.png', transparent=True, bbox_inches='tight')
    # plt.savefig('fig5.png', transparent=True)

    
def plotPeaks_4(extractPeaks):
    fig, axs = plt.subplots(1, len(extractPeaks), figsize=(8, 4), sharey=True)
    # fig, axs = plt.subplots(1, 6, sharey=True , figsize=(8, 4))
    
    plt.figure(6)
    # plt.clf()

    for i in range(len(extractPeaks)):
        axs[i].plot(np.arange(0,extractPeaks[i].shape[0])/rate, extractPeaks[i], color = '#1a9988')
        axs[i].set_xlabel('time(sec)')
        axs[i].set_ylabel('amp')
        axs[i].set_ylim([-1, 1])
        axs[i].title.set_text('Peak ' + str(i + 1))
    fig.suptitle('individual peaks - sound extraction')
    # plt.tight_layout()
    # fig.savefig('demo.png', transparent=True)
    # plt.savefig('fig6.png', transparent=True)
    



def psd_4(extractPeaks):

    sumPSDs = []
    fig, axs = plt.subplots(1, len(extractPeaks), figsize=(8, 4), sharey=True)
    # fig, axs = plt.subplots(1, 6, sharey=True , figsize=(8, 4))
    plt.figure(7)
    
    for i in range(len(extractPeaks)):
        #calculate the psd and sum them
        f, Pxx_den = scipy.signal.welch(extractPeaks[i], rate, nperseg=256)
        sumPSD = np.sum(Pxx_den)
        sumPSDs.append(sumPSD)
        #plot psd
        axs[i].semilogy(f, Pxx_den, color = '#1a9988',  marker='o')
        axs[i].set_xlabel('Hz')
        axs[i].set_ylabel('PSD')
        axs[i].set_ylim([10e-9, 10e-3])
        axs[i].set_xlim([2000, 5000])
        axs[i].title.set_text('Peak ' + str(i + 1))
    fig.suptitle('PSD of each peak')
    # plt.tight_layout()
    # fig.savefig('fig7.png', transparent=True)
    return sumPSDs

def plotEnergy(sumPSDs):
    #create labels
    #define threshold
    threshold = 0.7* np.max(sumPSDs)
    label = []
    for i in range(len(sumPSDs)):
        name = 'Peak ' + str(i+1)
        label.append(name)
    plt.figure(8)
    plt.clf()
    index = np.arange(len(label))
    barList = plt.bar(index, sumPSDs, color = '#1a9988')

    plt.plot(index, np.repeat(threshold, index.shape), color = 'r', linestyle = ':', label = 'threshold')
    plt.ylabel('Energey of peaks (sum of psd)')
    plt.xticks(index, label,rotation=0)
    plt.title('Energy of each peak')
    plt.tight_layout()
    #set bar color based of threshold
    #collect valid peak indexes
    ind = []
    for i in range(len(sumPSDs)):
        if sumPSDs[i] < threshold:
            barList[i].set_color('r')
            # barList[i].legend('reject peak')
        else:
            barList[i].set_color('#1a9988')
            # barList[i].legend('accept peak')
            ind.append(i)
    plt.legend()
    return ind


def plotEnergyRewrite(sumPSDs):
    #create labels
    #define threshold
    threshold = 0.7* np.max(sumPSDs)
    label = []
    for i in range(len(sumPSDs)):
        name = 'Peak ' + str(i+1)
        label.append(name)

    plt.figure(8)
    plt.clf()
    index = np.arange(len(label))

    ax1 = plt.subplot(111)

    ind = []    #accpted index
    for j in range(len(sumPSDs)):
        #select color and label
        if sumPSDs[j] < threshold:
            clr = '#ff0000'
            lbl = 'not peck'
        else:
             clr = '#1a9988'
             lbl = 'peak '
             ind.append(j)
        ax1.bar(index[j], sumPSDs[j], width=0.8, align='center', color=clr,  label=lbl if j <= 1 else "")

    ax1.set_xticks(index)
    ax1.set_xticklabels([label[i] for i in index])
    ax1.legend()

    plt.plot(index, np.repeat(threshold, index.shape), color = 'r', linestyle = ':', label = 'threshold')
    plt.ylabel('Energey of peaks (sum of psd)')
    plt.title('Energy of each peak with adaptive_threshold = 0.7*maxEnergy')
    plt.tight_layout()
    #set bar color based of threshold
    #collect valid peak indexes

    plt.legend()
    # plt.savefig('fig8.png', transparent=True)
    return ind



def plotPeaks(extractPeaks, validPeaksIndexes):
    fig, axs = plt.subplots(1, len(validPeaksIndexes), sharey=True, figsize = (6, 4))
    # fig, axs = plt.subplots(1, 6, sharey=True , figsize=(8, 4))
    
    plt.figure(9)
    # plt.clf()

    for i, index in enumerate(validPeaksIndexes):
        axs[i].plot(np.arange(0,extractPeaks[index].shape[0])/rate, extractPeaks[index], color = '#1a9988')
        axs[i].set_xlabel('time(sec)')
        axs[i].set_ylabel('amp')
        axs[i].set_ylim([-1, 1])
        axs[i].title.set_text('Peck ' + str(i + 1))
    fig.suptitle('Valid Peaks (Pecks)')
    # plt.tight_layout()
    fig.savefig('fig9.png', transparent=True)



bandpass_1()
hilbert_2()
filter_3()
extractPeaks = splitPeaks(signal, boolLoc)
plotPeaks_4(extractPeaks)
sumPSDs = psd_4(extractPeaks)
# print(sumPSDs)
validPeaksIndexes = plotEnergyRewrite(sumPSDs)

plotPeaks(extractPeaks, validPeaksIndexes)

plt.show()