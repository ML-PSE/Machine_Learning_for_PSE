##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                         RF quadratic fitting
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import
import numpy as np
np.random.seed(1)

#%% generate data
x = np.linspace(-1, 1, 50)[:, None]
y = x*x + 0.25 + np.random.normal(0, 0.15, (50,1))

#%% plot raw data
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(x, y, edgecolor="black", c="darkorange")
plt.xlabel('x'), plt.ylabel('y')

#%% fit RF model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=20).fit(x, y)

#%% predict and plot
y_pred = model.predict(x)

plt.figure()
plt.scatter(x, y, edgecolor="black", c="darkorange", label="raw data")
plt.plot(x, y_pred, 'yellowgreen', label="predictions")
plt.xlabel('x'), plt.ylabel('y')
plt.legend()

#%% get predictions from constituent trees and plot
tree_list = model.estimators_
y_pred_tree1 = tree_list[5].predict(x)
y_pred_tree2 = tree_list[15].predict(x)


plt.figure()
plt.scatter(x, y, edgecolor="black", c="darkorange", label="raw data")
plt.plot(x, y_pred_tree1, 'red', alpha=0.5, label="DT6 predictions")
plt.plot(x, y_pred_tree2, 'blue', alpha=0.5, label="DT16 predictions")
plt.xlabel('x'), plt.ylabel('y')
plt.legend()

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                  Impact of # of trees on validation error
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% generate validation data
np.random.seed(2)

x_val = np.linspace(-1, 1, 50)[:, None]
y_val = x_val*x_val + 0.25 + np.random.normal(0, 0.15, (50,1))

#%% plot raw validation data
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(x_val, y_val, edgecolor="black", c="darkorange")
plt.xlabel('x'), plt.ylabel('y')

#%% fit multiple RFs with # of trees ranging from 10 to 500
from sklearn.metrics import mean_squared_error as mse

val_errors = []
n_tree_grid = np.arange(2,250,5)
for n_tree in n_tree_grid:
    model = RandomForestRegressor(n_estimators=n_tree).fit(x, y)
    y_val_pred = model.predict(x_val)
    val_errors.append(mse(y_val, y_val_pred))

#%% plot validation errors
plt.figure()
plt.plot(n_tree_grid, val_errors, 'm')  
plt.xlabel('# of trees'), plt.ylabel('Validation MSE')
    






