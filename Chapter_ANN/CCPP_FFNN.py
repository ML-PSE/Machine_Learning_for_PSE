##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                         FFNN modeling of CCPP
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

#%% read data
data = pd.read_excel('Folds5x2_pp.xlsx', usecols = 'A:E').values
X = data[:,0:4]
y = data[:,4][:,np.newaxis]

#%% separate train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

#%% scale data
from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          Define & Fit FFNN model
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import Keras libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#%% define model
model = Sequential()
model.add(Dense(8, activation='relu', kernel_initializer='he_normal', input_shape=(4,))) # 8 neurons in 1st hidden layer; this hidden layer accepts data from a 4 dimensional input
model.add(Dense(5, activation='relu', kernel_initializer='he_normal')) # 5 neurons in 2nd layer
model.add(Dense(1)) # output layer

#%% compile model
model.compile(loss='mse', optimizer='Adam') # mean-squared error is to be minimized

#%% fit model
model.fit(X_train_scaled, y_train_scaled, epochs=25, batch_size=50)

#%% predict y_test
y_test_scaled_pred = model.predict(X_test_scaled)
y_test_pred = y_scaler.inverse_transform(y_test_scaled_pred)

plt.figure()
plt.plot(y_test, y_test_pred, '*')
plt.xlabel('y_test')
plt.ylabel('y_test_pred')

#%% metrics
from sklearn.metrics import r2_score
print('R2:', r2_score(y_test, y_test_pred))

#%% model summary
model.summary()