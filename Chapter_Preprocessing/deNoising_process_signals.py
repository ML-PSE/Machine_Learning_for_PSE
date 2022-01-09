##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    De-noising Process Signals
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% read data
import numpy as np
noisy_signal = np.loadtxt('noisy_flow_signal.csv', delimiter=',')

#%% SMA filter
import pandas as pd

windowSize = 15
smoothed_signal_MA = pd.DataFrame(noisy_signal).rolling(windowSize).mean().values

#%% SG filter
from scipy.signal import savgol_filter

smoothed_signal_SG = savgol_filter(noisy_signal, window_length = 15, polyorder = 2)

#%% plots
from matplotlib import pyplot as plt

plt.figure(figsize=(11,3))
plt.plot(noisy_signal, alpha=0.3, label='Noisy signal')
plt.plot(smoothed_signal_MA, color='m', label='SMA smoothed signal')
plt.plot(smoothed_signal_SG, color='orange', label='SG smoothed signal')
plt.xlabel('Sample #'), plt.ylabel('Value')
plt.legend()