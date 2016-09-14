import numpy as np
import soundfile as sf
import sidekit as skk
import os
import sys
import multiprocessing
import logging
import htkmfc as hhtk

from matplotlib import  pyplot as plt

print('Start')
signal, samplerate = sf.read('sw02289.sph')


Time=np.linspace(0, len(signal)/samplerate, num=len(signal))

# plt.figure(1)
# plt.title('Signal Wave...')
# plt.plot(Time,signal)
# plt.show()

# datahtk = np.reshape(signal[:,0], (1,lala))

a = skk.mfcc(signal[:,0])

ubm=skk.Mixture();

#x= skk.FeaturesExtractor(feature_filename_structure='sw02289.sph',shift=0.01,sampling_frequency=16000, window_size=0.025);


f, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.plot(a[0])
ax2.plot(a[1])
plt.show()