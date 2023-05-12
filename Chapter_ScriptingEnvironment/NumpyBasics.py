##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    Numpy Basics
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# create a 2D array
import numpy as np

arr2D = np.array([[1,4,6],[2,5,7]]) 

# getting information about arr2D
print(arr2D.size) # returns 6, the no. of items
print(arr2D.ndim) # returns 2, the no. of dimensions
print(arr2D.shape) # returns tuple(2,3) corresponding to 2 rows & 3 columns

# create a 1D array
arr1D = np.array([1,4,6]) 

# getting information about arr1D
print(arr1D.size) # returns 3, the no. of items
print(arr1D.ndim) # returns 1, the no. of dimensions
print(arr1D.shape) # returns tuple(3,) corresponding to 3 items

#%% creating numpy arrays
# creating sequence of numbers
arr1 = np.arange(3, 6) # same as Python range function; results in array([3,4,5])
arr2 = np.arange(3, 9, 2) # the 3rd argument defines the step size; results in array([3,5,7])
arr3 = np.linspace(1,7,3) # creates evenly spaced 3 values from 1 to 7; results in array([1,4,7])

# creating special arrays
arr4 = np.ones((2,1)) # array of shape (2,1) with all items as 1
arr5 = np.zeros((2,2)) # all items as zero; often used as placeholder array at beginning of script
arr6 = np.eye(2) # diagonal items as 1

# adding axis to existing arrays (e.g., converting 1D array to 2D array)
print(arr1[:, np.newaxis])
arr7 = arr1[:, None] # same as above

# combining / stacking arrays
print(np.hstack((arr1, arr2))) # horizontally stacks passed arrays
print(np.vstack((arr1, arr2))) # vertically stacks passed arrays
print(np.hstack((arr5,arr4))) # array 4 added as a column into arr5
print(np.vstack((arr5,arr6))) # rows of array 6 added onto arr5

#%% basic numpy functions
print(arr2D.sum(axis=0))
print(arr2D.sum(axis=1))

#%% indexing arrays
# accessing individual items
print(arr2D[1,2]) # returns 7

# slicing
arr8 = np.arange(10).reshape((2,5)) # rearrange the 1D array into shape (2,5)
print((arr8[0:1,1:3]))
print((arr8[0,1:3])) # note that a 1D array is returned here instead of the 2D array above

# accessing entire row or column
print(arr8[1]) # returns 2nd row as array([5,6,7,8,9]); same as arr8[1,:]
print(arr8[:, 4]) # returns items of 5th column as a 1D array 

# extract a subarray from arr8 and modify it
arr8_sub = arr8[:, :2] # columns 0 and 1 from all rows
arr8_sub[1, 1] = 1000
print(arr8) # arr8 gets modified as well!! 

# use copy method for a separate copy
arr8 = np.arange(10).reshape((2,5))
arr8_sub2 = arr8[:, :2].copy()
arr8_sub2[1, 1] = 100
print(arr8)

# Fancy indexing
# combination of simple and fancy indexing
arr8_sub3 = arr8[:, [0, 1]] # note how columns are indexed via a list
arr8_sub3[1, 1] = 100 # arr8_sub3 becomes same as arr8_sub2 but arr8 is not modified here
print(arr8)

# use boolean mask to select subarray
arr8_sub4 = arr8[arr8 > 5] # returns array([6,7,8,9]), i.e., all values > 5
arr8_sub4[0] = 0 # again, arr8 is not affected
print(arr8)

#%% vectorized operations
vec1 = np.array([1,2,3,4])
vec2 = np.array([5,6,7,8])
vec_sum = vec1 + vec2 # returns array([6,8,10,12]); no need to loop through index 0 to 3

# slightly more complex operation (computing distance between vectors)
vec_distance = np.sqrt(np.sum((vec1 - vec2)**2)) # vec_distance = 8.0
