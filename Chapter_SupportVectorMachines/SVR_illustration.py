##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                         SVR quadratic fitting
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import
import numpy as np
np.random.seed(1)

#%% generate data
x = np.linspace(-1, 1, 50)[:, None]
y = x*x + 0.25
y = y + np.random.normal(0, 0.15, (50,1))

#%% plot
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(x,y,edgecolors='k', alpha=0.8)
plt.xlabel('x'), plt.ylabel('y')

#%% fit SVR model
from sklearn.svm import SVR

epsilon = 0.1
model = SVR(gamma=0.5, C=10, epsilon=epsilon)
model.fit(x, y)

#%% predict
xx = np.linspace(-1, 1, 200)[:, None]
yy_predicted = model.predict(xx)
yy_epsilon_tube_upper = yy_predicted + epsilon
yy_epsilon_tube_lower = yy_predicted - epsilon

#%% get support vectors
x_SVs = model.support_vectors_
y_SVs = y[model.support_]

#%% plot
plt.figure()
plt.scatter(x,y,edgecolors='k', alpha=0.8)
plt.plot(xx, yy_predicted, 'r')
plt.plot(xx, yy_epsilon_tube_upper, '--g')
plt.plot(xx, yy_epsilon_tube_lower, '--g')
plt.scatter(x_SVs, y_SVs, s=200, linewidth=2, edgecolors='m', alpha=0.15)
plt.xlabel('x'), plt.ylabel('y')




