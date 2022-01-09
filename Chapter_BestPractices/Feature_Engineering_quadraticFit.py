##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    Feature Engineering (quadratic fit via linear model)
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% read data
import numpy as np
data = np.loadtxt('quadratic_raw_data.csv', delimiter=',')
x = data[:,0,None]; y = data[:,1,None]

# plot
import matplotlib.pyplot as plt
plt.figure()
plt.plot(x,y, 'o')
plt.xlabel('x'), plt.ylabel('y')

#%% generate quadratic features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(x) # X_poly: 1st column is x, 2nd column is x^2 

#%% scale model inputs
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X_poly) 

#%% linear fit & predict
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_poly, y)
y_predicted = model.predict(X_poly)

#%% check predictions
# plot
plt.figure()
plt.plot(x,y, 'o', label='raw data')
plt.plot(x,y_predicted, label='quadratic fit')
plt.legend()
plt.xlabel('x'), plt.ylabel('y')

# accuracy
from sklearn.metrics import r2_score
print('Fit accuracy = ', r2_score(y, y_predicted))