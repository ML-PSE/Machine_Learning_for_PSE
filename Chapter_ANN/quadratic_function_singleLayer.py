##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                         FFNN modeling of y = x*x
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np
import matplotlib.pyplot as plt

#%% random number seed for result reproducibility 
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)

#%% generate data
x = np.linspace(-1,1,500)
y = x*x 
plt.plot(x,y)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          Define & Fit FFNN model
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import Keras libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

#%% define model
n_nodes = 5

model = Sequential()
model.add(Dense(n_nodes, activation='relu', input_shape=(1,)))
model.add(Dense(1))

#%% compile model
model.compile(loss='mse', optimizer=Adam(learning_rate=0.05))

#%% fit model
history = model.fit(x, y, epochs=400, batch_size=50)

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.plot(history.history['loss'], label='train')
plt.show()

#%% predict y_test
y_pred = model.predict(x)

plt.figure()
plt.plot(x, y, '--b', label='y=x^2')
plt.plot(x, y_pred, '--r', label='Approximation')
plt.xlabel('x')
plt.title('y_pred vs y')
plt.legend()

plt.figure()
plt.plot(y_pred, 'r')
plt.title('y_pred')

#%% metrics
from sklearn.metrics import r2_score
print('R2:', r2_score(y, y_pred))

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          inner layer activations
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import tensorflow.keras.backend as K
activations = []
for layer in model.layers:
    keras_function = K.function([model.input], [layer.output])
    activations.append(keras_function(x))

#%% plot activations
layer1_activations = activations[0][0]
for node in range(n_nodes):
    plt.figure()
    plt.plot(x, layer1_activations[:,node])
    plt.title('node ' + str(node+1) + ' activation')