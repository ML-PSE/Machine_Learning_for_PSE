##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                     Exploring aircraft engine data
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#%% read data
# training
train_df = pd.read_csv('PM_train.txt', sep=" ", header=None)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True) # last two columns are blank
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']

# test 
test_df = pd.read_csv('PM_test.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']

# actual RUL for each engine-id in the test data
truth_df = pd.read_csv('PM_truth.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                     exploratory graphs (training)
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# get all sensor data for an engine ID
engineID = 1
engineDataAll = train_df.loc[train_df['id'] == engineID]
engineDataSensor = engineDataAll.iloc[:, 5:]

# normalize
scalar = StandardScaler()
engineDataSensor_scaled = scalar.fit_transform(engineDataSensor.values)

# plot all sensor data for an engine ID
plt.figure()
plt.plot(engineDataSensor_scaled)
plt.xlabel('Engine cycle')
plt.ylabel('Scaled sensor values')
plt.title('Training sensor Data for engineID ' + str(engineID))
plt.box(False)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                     exploratory graphs (test)
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# get all sensor data for an engine ID
engineID = 90
engineDataAll = test_df.loc[test_df['id'] == engineID]
engineDataSensor = engineDataAll.iloc[:, 5:]

# normalize
scalar = StandardScaler()
engineDataSensor_scaled = scalar.fit_transform(engineDataSensor.values)

# plot all sensor data for an engine ID
plt.figure()
plt.plot(engineDataSensor_scaled)
plt.xlabel('Engine cycle')
plt.ylabel('Scaled sensor values')
plt.title('Test sensor Data for engineID ' + str(engineID))
plt.box(False)
