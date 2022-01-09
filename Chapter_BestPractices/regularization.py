##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    Regularization to prevent overfitting
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% read data
import numpy as np
data = np.loadtxt('quadratic_raw_data.csv', delimiter=',')
x = data[:,0,None]; y = data[:,1,None]

# separate training data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    Ordinary Least Squares Regression
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% create pipeline for quadratic fit via OLS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pipe_OLS = Pipeline([('poly', PolynomialFeatures(degree=10, include_bias=False)),
                 ('scaler', StandardScaler()),
                 ('model', LinearRegression())])

#%% fit pipeline and predict
pipe_OLS.fit(x_train, y_train)
y_predicted_train_OLS = pipe_OLS.predict(x_train)
y_predicted_test_OLS = pipe_OLS.predict(x_test)

#%% performance metrics
from sklearn.metrics import mean_squared_error as mse

print('OLS Training metric (mse) = ', mse(y_train, y_predicted_train_OLS))
print('OLS Test metric (mse) = ', mse(y_test, y_predicted_test_OLS))

#%% plot predictions
y_predicted_OLS = pipe_OLS.predict(x)

from matplotlib import pyplot as plt
plt.figure()
plt.plot(x_train,y_train, 'bo', label='raw training data')
plt.plot(x_test,y_test, 'ro', label='raw test data')
plt.plot(x,y_predicted_OLS, color='orange', label='OLS fit')
plt.legend()
plt.xlabel('x'), plt.ylabel('y')

#%% print coefficients
print(pipe_OLS['model'].coef_)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    Ridge Regression
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% create pipeline for quadratic fit via ridge model 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipe_L2 = Pipeline([('poly', PolynomialFeatures(degree=10,include_bias=False)),
                    ('scaler', StandardScaler()),
                    ('model', Ridge(alpha=0.1))])

#%% fit pipeline and predict
pipe_L2.fit(x_train, y_train)
y_predicted_train_L2 = pipe_L2.predict(x_train)
y_predicted_test_L2 = pipe_L2.predict(x_test)

#%% performance metrics
from sklearn.metrics import mean_squared_error as mse

print('L2 Training metric (mse) = ', mse(y_train, y_predicted_train_L2))
print('L2 Test metric (mse) = ', mse(y_test, y_predicted_test_L2))

#%% plot predictions
y_predicted_L2 = pipe_L2.predict(x)

plt.figure()
plt.plot(x_train,y_train, 'bo', label='raw training data')
plt.plot(x_test,y_test, 'ro', label='raw test data')
plt.plot(x,y_predicted_L2, color='orange', label='Ridge fit')
plt.legend()
plt.xlabel('x'), plt.ylabel('y')

#%% print coefficients
print(pipe_L2['model'].coef_)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    Lasso Regression
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% create pipeline for quadratic fit via ridge model 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

pipe_L1 = Pipeline([('poly', PolynomialFeatures(degree=10,include_bias=False)),
                    ('scaler', StandardScaler()),
                    ('model', Lasso(alpha=0.1))])

#%% fit pipeline and predict
pipe_L1.fit(x_train, y_train)
y_predicted_train_L1 = pipe_L1.predict(x_train)
y_predicted_test_L1 = pipe_L1.predict(x_test)

#%% performance metrics
from sklearn.metrics import mean_squared_error as mse

print('L1 Training metric (mse) = ', mse(y_train, y_predicted_train_L1))
print('L1 Test metric (mse) = ', mse(y_test, y_predicted_test_L1))

#%% plot predictions
y_predicted_L1 = pipe_L1.predict(x)

plt.figure()
plt.plot(x_train,y_train, 'bo', label='raw training data')
plt.plot(x_test,y_test, 'ro', label='raw test data')
plt.plot(x,y_predicted_L1, color='orange', label='Lasso fit')
plt.legend()
plt.xlabel('x'), plt.ylabel('y')

#%% print coefficients
print(pipe_L1['model'].coef_)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                        Combined plot
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plt.figure()
plt.plot(x_train,y_train, 'bo', label='raw training data')
plt.plot(x_test,y_test, 'ro', label='raw test data')
plt.plot(x,y_predicted_OLS, color='orange', label='OLS fit')
plt.plot(x,y_predicted_L2, label='Ridge fit')
plt.plot(x,y_predicted_L1, label='Lasso fit')
plt.legend()
plt.xlabel('x'), plt.ylabel('y')