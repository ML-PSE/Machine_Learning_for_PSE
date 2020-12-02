##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          Exploration of debutanizer data
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np
import matplotlib.pyplot as plt

#%% read data
data = np.loadtxt('debutanizer_data.txt', skiprows=5)

#%% plot each variable
plt.figure()
plt.plot(data[:,0])
plt.ylabel('top Temperature')
plt.xlabel('samples')
plt.xlim((0,2500))

plt.figure()
plt.plot(data[:,1])
plt.ylabel('top Pressure')
plt.xlabel('samples')
plt.xlim((0,2500))

plt.figure()
plt.plot(data[:,2])
plt.ylabel('reflux flow')
plt.xlabel('samples')
plt.xlim((0,2500))

plt.figure()
plt.plot(data[:,3])
plt.ylabel('flow to next process')
plt.xlabel('samples')
plt.xlim((0,2500))

plt.figure()
plt.plot(data[:,4])
plt.ylabel('6th tray Temperature')
plt.xlabel('samples')
plt.xlim((0,2500))

plt.figure()
plt.plot(data[:,5])
plt.ylabel('bottom Temperature 1')
plt.xlabel('samples')
plt.xlim((0,2500))

plt.figure()
plt.plot(data[:,6])
plt.ylabel('bottom Temperature 2')
plt.xlabel('samples')
plt.xlim((0,2500))

plt.figure()
plt.plot(data[:,7])
plt.ylabel('C4 content')
plt.xlabel('samples')
plt.xlim((0,2500))