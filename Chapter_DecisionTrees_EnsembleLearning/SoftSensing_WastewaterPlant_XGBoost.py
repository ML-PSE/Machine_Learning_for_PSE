##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##       Soft sensing via XGBoost on UCI Wastewater Treatment Plant data
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% read data
import pandas as pd

data_raw = pd.read_csv('water-treatment.data', header=None,na_values="?" ) # dataset uses '?' to denote missing value
X_raw = data_raw.iloc[:,1:23]
y_raw = data_raw.iloc[:,29]

#%% handle missing data
# generate a dataframe from inputs dataframe and output series
data = pd.concat([X_raw, y_raw], axis=1)

# check for presence of missing values
print(data.info())

# remove rows with missing data
data.dropna(axis=0, how='any', inplace=True)

print('Number of samples remaining:', data.shape[0])

#%% separate inputs and output
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

#%% plot 1st input and output to show variability
from matplotlib import pyplot as plt
plt.figure()
plt.plot(X.iloc[:,0].values, color='brown', linestyle = ':', marker='.', linewidth=0.5, markeredgecolor = 'k')
plt.xlabel('Sample #')
plt.ylabel('Input flow to plant')

plt.figure()
plt.plot(X.iloc[:,8].values, color='brown', linestyle = ':', marker='.', linewidth=0.5, markeredgecolor = 'k')
plt.xlabel('Sample #')
plt.ylabel('Input conductivity to plant')

plt.figure()
plt.plot(y.values, color='navy', linestyle = ':', marker='.', linewidth=0.5, markeredgecolor = 'k')
plt.xlabel('Sample #')
plt.ylabel('Output Conductivity')

#%% separate fitting, validation, and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 100)

#%% fit XGBoost model
import xgboost 
model = xgboost.XGBRegressor(max_depth=3, learning_rate=0.1, random_state=100)
model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], early_stopping_rounds=2)

#%% predict and plot
y_train_predicted = model.predict(X_train)
y_test_predicted = model.predict(X_test)

plt.figure()
plt.plot(y_train, y_train_predicted, '.', markeredgecolor='k', markeredgewidth=0.5, ms=9)
plt.plot(y_train, y_train, '-r', linewidth=0.5)
plt.xlabel('raw training data')
plt.ylabel('prediction')

plt.figure()
plt.plot(y_test, y_test_predicted, '.', markeredgecolor='k', markeredgewidth=0.5, ms=9)
plt.plot(y_test, y_test, '-r', linewidth=0.5)
plt.xlabel('raw test data')
plt.ylabel('prediction')

#%% check training vs test accuracy
from sklearn.metrics import r2_score
print('Accuracy over training data: ', r2_score(y_train, y_train_predicted))
print('Accuracy over test data: ', r2_score(y_test, y_test_predicted))



