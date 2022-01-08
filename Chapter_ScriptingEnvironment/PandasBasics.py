##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    Pandas Basics
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# create a series (1D structure)
import pandas as pd

data = [10,8,6]
s = pd.Series(data) # can pass numpy array as well
print(s)

# create a dataframe
data = [[1,10],[1,8],[1,6]]
df = pd.DataFrame(data, columns=['id', 'value'])
print(df)

# dataframe from series
s2 = pd.Series([1,1,1])
df = pd.DataFrame({'id':s2, 'value':s})
print(df)

#%% data access
# column(s) selection
print(df['id']) # returns column 'id' as a series
print(df.id) # same as above
print(df[['id']]) # returns specified columns in the list as a dataframe

# row selection
df.index = [100, 101, 102] # changing row indices from [0,1,2] to [100,101,102]
print(df)
print(df.loc[101]) # returns 2nd row as a series; can provide a list for multiple rows selection
print(df.iloc[1]) # integer location-based selection; same result as above

# individual item selection
print(df.loc[101, 'value']) # returns 8
print(df.iloc[1, 1]) # same as above

#%% data aggregation exanple
# create another dataframe using df
df2 = df.copy()
df2.id = 2 # make all items in column 'id' as 2
df2.value *= 4 # multiply all items in column 'value' by 4
print(df2)

# combine df and df2
df3 = df.append(df2) # a new object is retuned unlike Python’s append function
print(df3)

# id-based mean values computation
print(df3.groupby('id').mean()) # returns a dataframe

#%% file I/O
# reading from excel and csv files
dataset1 = pd.read_excel('filename.xlsx') # several parameter  options are available to customize what data is read
dataset2 = pd.read_csv('filename.xlsx')
