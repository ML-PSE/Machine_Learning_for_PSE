##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          exploration of CCPP data
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

#%% read data
data = pd.read_excel('Folds5x2_pp.xlsx', usecols = 'A:E').values
X = data[:,0:4]
y = data[:,4][:,np.newaxis]

#%% plot input vs output for each input
plt.figure()
plt.plot(X[:,0], y, '*')
plt.title('AT vs EP')

plt.figure()
plt.plot(X[:,1], y, '*')
plt.title('V vs EP')

plt.figure()
plt.plot(X[:,2], y, '*')
plt.title('AP vs EP')

plt.figure()
plt.plot(X[:,3], y, '*')
plt.title('RH vs EP')