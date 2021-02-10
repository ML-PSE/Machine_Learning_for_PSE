##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                RNN-based System Identification of a SISO Heater
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

#%% read data
data = pd.read_csv('TCLab_train_data.txt')
heaterPower = data[['Q1']].values
temperature = data[['T1']].values

#%% plot data
plt.plot(temperature, 'k')
plt.ylabel('Temperature')
plt.xlabel('Time (sec)')

plt.figure()
plt.plot(heaterPower)
plt.ylabel('Heater Power')
plt.xlabel('Time (sec)')

#%% decide model input-outputs and scale data
from sklearn.preprocessing import StandardScaler

X = data[['T1','Q1']].values
y = data[['T1']].values

X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          re-arrage data with time steps
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# rearrange X data into (# sequence samples, # time steps, # features) form
nTimeSteps = 70
X_train_sequence = []
y_train_sequence = []

for sample in range(nTimeSteps, X_scaled.shape[0]):
    X_train_sequence.append(X_scaled[sample-nTimeSteps:sample,:])
    y_train_sequence.append(y_scaled[sample])

# X conversion: convert list of (time steps, features) arrays into (samples, time steps, features) array 
X_train_sequence, y_train_sequence = np.array(X_train_sequence), np.array(y_train_sequence) 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          fit LSTM model
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# import Keras libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import regularizers

#%% define model
model = Sequential()
model.add(LSTM(units=25, kernel_regularizer=regularizers.L1(0.001), input_shape=(nTimeSteps,2)))
model.add(Dense(units=1))

#%% model summary
model.summary()

#%% compile model
model.compile(loss='mse', optimizer='Adam')

#%% fit model with early stopping
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_sequence, y_train_sequence, epochs=100, batch_size=250, validation_split=0.3, callbacks=[es])

#%% plot validation curve
plt.figure()
plt.title('Validation Curves')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.grid()
plt.show()

#%% check fit on training data
y_train_sequence_pred = model.predict(X_train_sequence) 
y_measured = y_scaler.inverse_transform(y_train_sequence)
y_pred =  y_scaler.inverse_transform(y_train_sequence_pred)

#%% plot
plt.figure()
plt.plot(y_pred, 'r-', label='LSTM prediction')
plt.plot(y_measured, 'k--', label='raw measurements')
plt.ylabel('Temperature (°C)')
plt.xlabel('Time (sec)')
plt.legend()
plt.title('Training data')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                         predict for test data
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% read test data
data_test = pd.read_csv('TCLab_test_data.txt')
X_test = data_test[['T1','Q1']].values
y_test = data_test[['T1']].values

#%% scale data
X_test_scaled = X_scaler.transform(X_test)
y_test_scaled = y_scaler.transform(y_test)

#%% re-arrange data into sequence form
X_test_sequence = []
y_test_sequence = []

for sample in range(nTimeSteps, X_test_scaled.shape[0]):
    X_test_sequence.append(X_test_scaled[sample-nTimeSteps:sample,:])
    y_test_sequence.append(y_test_scaled[sample])

X_test_sequence, y_test_sequence = np.array(X_test_sequence), np.array(y_test_sequence)

#%% predict y_test
y_test_sequence_pred = model.predict(X_test_sequence) 
y_test_pred =  y_scaler.inverse_transform(y_test_sequence_pred)

#%% plot
nRows_test = y_test.shape[0]

# temperature
plt.figure()
plt.plot(np.arange(nTimeSteps, nRows_test), y_test_pred, 'r-', label='LSTM prediction')
plt.plot(np.arange(nRows_test), y_test, 'k--', label='raw measurements')
plt.ylabel('Temperature (°C)')
plt.xlabel('Time (sec)')
plt.legend()
plt.title('Test data')

# heater power
plt.figure()
plt.plot(X_test[:,1])
plt.ylabel('Heater Power')
plt.xlabel('Time (sec)')