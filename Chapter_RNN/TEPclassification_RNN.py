##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    RNN-based TEP fault classification
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np
import matplotlib.pyplot as plt

#%% read data
import pyreadr
fault_free_training_data = pyreadr.read_r('TEP_FaultFree_Training.RData')['fault_free_training'] # pandas dataframe
fault_free_testing_data = pyreadr.read_r('TEP_FaultFree_Testing.RData')['fault_free_testing']
faulty_training_data = pyreadr.read_r('TEP_Faulty_Training.RData')['faulty_training']
faulty_testing_data = pyreadr.read_r('TEP_Faulty_Testing.RData')['faulty_testing']

#%% check data
fault_free_training_data.head()

#%% remove fault 3,9,15 data from faulty dataset
faulty_training_data = faulty_training_data[faulty_training_data['faultNumber'] != 3]
faulty_training_data = faulty_training_data[faulty_training_data['faultNumber'] != 9]
faulty_training_data = faulty_training_data[faulty_training_data['faultNumber'] != 15]

faulty_testing_data = faulty_testing_data[faulty_testing_data['faultNumber'] != 3]
faulty_testing_data = faulty_testing_data[faulty_testing_data['faultNumber'] != 9]
faulty_testing_data = faulty_testing_data[faulty_testing_data['faultNumber'] != 15]

#%% separate validation dataset out of training dataset and create imbalanced faulty dataset
fault_free_validation_data = fault_free_training_data[fault_free_training_data['simulationRun'] > 400]
fault_free_training_data = fault_free_training_data[fault_free_training_data['simulationRun'] <= 400]
faulty_validation_data = faulty_training_data[faulty_training_data['simulationRun'] > 490] 
faulty_training_data = faulty_training_data[faulty_training_data['simulationRun'] <= 50]

#%% convert to numpy
fault_free_training_data = fault_free_training_data.values
fault_free_validation_data = fault_free_validation_data.values
fault_free_testing_data = fault_free_testing_data.values
faulty_training_data = faulty_training_data.values
faulty_validation_data = faulty_validation_data.values
faulty_testing_data = faulty_testing_data.values

#%% complete training, validation, and test datasets
training_data = np.vstack((fault_free_training_data,faulty_training_data))
validation_data = np.vstack((fault_free_validation_data,faulty_validation_data))
testing_data = np.vstack((fault_free_testing_data,faulty_testing_data))

#%% separate X and y data
X_train = training_data[:,3:]
X_val = validation_data[:,3:]
X_test = testing_data[:,3:]

y_train = training_data[:,0]
y_val = validation_data[:,0]
y_test = testing_data[:,0]

#%% scale data
from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_val_scaled = X_scaler.transform(X_val)
X_test_scaled = X_scaler.transform(X_test)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                         re-arrage data with time steps 
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# rearrange X data into (# sequence samples, # time steps, # features) form
nTimeStepsTrain = 500 # length of a simulation run in training data 
nTimeStepsTest = 960 # length of a simulation run in testing data 
X_train_sequence = []
y_train_sequence = []
X_val_sequence = []
y_val_sequence = []
X_test_sequence = []
y_test_sequence = []

for sample in range(0, X_train_scaled.shape[0], nTimeStepsTrain):
    X_train_sequence.append(X_train_scaled[sample:sample+nTimeStepsTrain,:])
    y_train_sequence.append(y_train[sample])
    
for sample in range(0, X_val_scaled.shape[0], nTimeStepsTrain):
    X_val_sequence.append(X_val_scaled[sample:sample+nTimeStepsTrain,:])
    y_val_sequence.append(y_val[sample])    

for sample in range(0, X_test_scaled.shape[0], nTimeStepsTest):
    X_test_sequence.append(X_test_scaled[sample:sample+nTimeStepsTest,:])
    y_test_sequence.append(y_test[sample])
    
X_train_sequence, y_train_sequence = np.array(X_train_sequence), np.array(y_train_sequence) 
X_val_sequence, y_val_sequence = np.array(X_val_sequence), np.array(y_val_sequence)
X_test_sequence, y_test_sequence = np.array(X_test_sequence), np.array(y_test_sequence)

#%% convert fault class labels to one-hot encoded form
from tensorflow.keras.utils import to_categorical
Y_train_sequence = to_categorical(y_train_sequence, num_classes=21)
Y_val_sequence = to_categorical(y_val_sequence, num_classes=21)
Y_test_sequence = to_categorical(y_test_sequence, num_classes=21) 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          fit LSTM model
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# import packages
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import regularizers

#%% define model
model = Sequential()
model.add(LSTM(units=128, kernel_regularizer=regularizers.L1(0.0001), return_sequences=True, input_shape=(nTimeStepsTrain,52)))
model.add(LSTM(units=64, kernel_regularizer=regularizers.L1(0.0001)))
model.add(Dense(21, activation='softmax'))

#%% model summary
model.summary()

#%% compile model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

#%% fit model with early stopping
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train_sequence, Y_train_sequence, epochs=100, batch_size=250, validation_data=(X_val_sequence,Y_val_sequence), callbacks=[es])

#%% plot validation curve
plt.figure()
plt.title('Validation Curves: Loss')
plt.xlabel('Epoch'), plt.ylabel('Loss')
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title('Validation Curves: Accuracy')
plt.xlabel('Epoch'), plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.grid()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                         evaluate model on test dataset
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# model performance on test data
resultsTest  = model.evaluate(X_test_sequence, Y_test_sequence) 
print("test accuracy:", resultsTest[1])

#%% generate confusion matrix
from sklearn.metrics import confusion_matrix

Y_test_sequence_pred = model.predict(X_test_sequence)
y_test_sequence_pred = np.argmax(Y_test_sequence_pred, axis = 1)
conf_matrix = confusion_matrix(y_test_sequence, y_test_sequence_pred, labels=list(range(21)))

#%% plot confusion matrix
import seaborn as sn

sn.set(font_scale=1.5) # for label size
sn.heatmap(conf_matrix, fmt='.0f', annot=True, cmap='Blues')
plt.ylabel('True Fault Class', fontsize=35)
plt.xlabel('Predicted Fault Class', fontsize=35)
