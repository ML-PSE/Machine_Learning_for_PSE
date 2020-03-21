# -*- coding: utf-8 -*-
"""

"""
#%% import required packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#%% fetch data
data = pd.read_excel('proc1a.xls', skiprows = 1,usecols = 'C:AI')

#%% separate train data
data_train = data.iloc[0:69,]

# %% augment training data
lag = 5
N = data_train.shape[0]
m = data_train.shape[1]

data_train_augmented = np.zeros((N-lag+1,lag*m))

for sample in range(lag, N):
    dataBlock = data_train.iloc[sample-lag:sample,:].values # converting from pandas dataframe to numpy array
    data_train_augmented[sample-lag,:] = np.reshape(dataBlock, (1,-1), order = 'F')
           
#%% scale data
scaler = StandardScaler()
data_train_augmented_normal = scaler.fit_transform(data_train_augmented)

#%% PCA
pca = PCA()
score_train = pca.fit_transform(data_train_augmented_normal)




