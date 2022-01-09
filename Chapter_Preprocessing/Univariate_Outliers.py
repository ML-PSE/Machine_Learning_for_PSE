##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    Univariate outlier detection
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% Generate oulier-infested data
import numpy as np

X = np.random.normal(40, 1, (1500,1))
X[200:300] = X[200:300] +8; X[1000:1150] = X[1000:1150] + 8

# plot
import matplotlib.pyplot as plt
plt.plot(X, '.-')
plt.xlabel('sample #'), plt.ylabel('variable measurement')
plt.title('Raw measurements')

#%% 3-sigma rule
# location & spread
mu = np.mean(X)
sigma = np.std(X)

# mean and std
print('Estimated mean = ', mu)
print('Estimated standard deviation = ', sigma)

# plot
plt.figure()
plt.plot(X, '.-', alpha=0.8, markeredgecolor='k', markeredgewidth=0.1, ms=3)
plt.hlines(mu, 0, 1500, colors='m', linestyles='dashdot', label='Mean') 
plt.hlines(mu+3*sigma, 0, 1500, colors='r', linestyles='dashdot', label='Upper bound') 
plt.hlines(mu-3*sigma, 0, 1500, colors='r', linestyles='dashed', label='Lower bound') 

plt.xlabel('sample #'), plt.ylabel('Variable measurement')
plt.xlim((0,1500))
plt.title('3-sigma bounds')
plt.legend(loc='upper right')

#%% hampel identifier
# compute median and MAD
from scipy import stats

median = np.median(X)
sigma_MAD = stats.median_absolute_deviation(X) # default scaling of 1.4826 is built-in

# median & sigma_MAD
print('Estimated robust location = ', median)
print('Estimated robust spread = ', sigma_MAD)

# plot
plt.figure()
plt.plot(X, '.-', alpha=0.8, markeredgecolor='k', markeredgewidth=0.1, ms=3)
plt.hlines(median, 0, 1500, colors='m', linestyles='dashdot', label='Mean') 
plt.hlines(median+3*sigma_MAD, 0, 1500, colors='r', linestyles='dashdot', label='Upper bound') 
plt.hlines(median-3*sigma_MAD, 0, 1500, colors='r', linestyles='dashed', label='Lower bound') 

plt.xlabel('sample #'), plt.ylabel('Variable measurement')
plt.xlim((0,1500))
plt.title('Hampel identifier bounds')
plt.legend(loc='upper right')