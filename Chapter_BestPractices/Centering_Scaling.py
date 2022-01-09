##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    Centering & Scaling
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% Standard scaling
import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.array([[ 1000, 0.01,  300],
              [ 1200,  0.06,  350], 
              [ 1500,  0.1, 320]])
scaler = StandardScaler().fit(X) # computes mean & std column-wise
X_scaled = scaler.transform(X) # transform using computed mean and std

# check mean = 0 and variance = 1 for every variable/column after scaling 
print(X_scaled.mean(axis=0)) # return 1D array of size(3,1)
print(X_scaled.std(axis=0)) # return 1D array of size(3,1)

# access mean and variance via object properties
print(scaler.mean_) # return 1D array of size(3,1)
print(scaler.var_) # return 1D array of size(3,1)

#%% Normalization
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() # create object
X_scaled = scaler.fit_transform(X) # fit & transform 

# check min = 0 and max = 1 for every variable/column after scaling 
print(X_scaled.min(axis=0))
print(X_scaled.max(axis=0))

# access min and max via object properties
print(scaler.data_min_)
print(scaler.data_max_)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    Robust Centering & Scaling
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% Generate oulier-infested data
X = np.random.normal(40, 1, (1500,1))
X[200:300] = X[200:300] +8; X[1000:1150] = X[1000:1150] + 8

# plot
import matplotlib.pyplot as plt
plt.plot(X, '.-')
plt.xlabel('sample #'), plt.ylabel('variable measurement')
plt.title('Raw measurements')

#%% Transform via standard scaling
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# mean and std
print('Estimated mean = ', scaler.mean_[0])
print('Estimated standard deviation = ', np.sqrt(scaler.var_[0]))

# plot
plt.figure()
plt.plot(X_scaled, '.-')
plt.xlabel('sample #'), plt.ylabel('scaled variable measurement')
plt.xlim((0,1500))
plt.title('Standard scaling')

#%% Transform via robust MAD scaling
# compute median and MAD
from scipy import stats
median = np.median(X)
MAD = stats.median_absolute_deviation(X)

# scale
X_scaled = (X - median)/MAD[0]

# median and MAD
print('Estimated robust location = ', median)
print('Estimated robust spread = ', MAD)

# plot
plt.figure()
plt.plot(X_scaled, '.-')
plt.xlabel('sample #'), plt.ylabel('scaled variable measurement')
plt.xlim((0,1500))
plt.title('Robust MAD scaling')

