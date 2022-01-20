##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          SVR model with polymer plant data
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np
import matplotlib.pyplot as plt

#%% read data
data = np.loadtxt('polymer.dat')
X = data[:,0:10]
Y = data[:,10:]
y = Y[:,2]

#%% fit SVR model
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

model = SVR(epsilon=0.01) # default epsilon = 0.1
param_grid = [{'gamma': np.linspace(0.1e-05,5,100), 'C': np.linspace(0.01,5000,100)}]
gs = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=10, verbose=2)

gs.fit(X, y)
print('Optimal hyperparameter:', gs.best_params_)

#%% predict using the best model
y_predicted_SVR = gs.predict(X) 

#%% plots of raw and predicted data
plt.figure()
plt.plot(y, y_predicted_SVR, '.', markeredgecolor='k', markeredgewidth=0.5, ms=9)
plt.plot(y, y, '-r', linewidth=0.5)
plt.xlabel('measured data'), plt.ylabel('predicted data ')

#%% metrics
from sklearn.metrics import r2_score
print('R2:', r2_score(y, y_predicted_SVR))