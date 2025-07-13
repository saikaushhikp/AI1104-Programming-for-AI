# importing modules

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

import statistics as stats

import random

from matplotlib import pyplot as plt


def mse_calculator(value, estimator):       # a calculator for computing (x-y)^2 and returning it
    myValue= np.mean((value-estimator) ** 2)
    return myValue

def mode_calc(arr):                     # a mode calculator as per the special requirement of the question 3.4 part 4
    
    frequency_dictionary={}
    for item in arr:
        if item in frequency_dictionary:
            frequency_dictionary[item]+=1
        else:
            frequency_dictionary[item]=1
            
    max_frequency=max(frequency_dictionary.values())
    modes=[]
    
    for key in frequency_dictionary.keys():
        if frequency_dictionary[key]==max_frequency:
            modes.append(key)
    
    if len(modes)>1:
        return random.choice(modes)
    elif len(modes)==1:
        return modes[0]


# Question 3.1

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"  # copying url for dataframe
abalone = pd.read_csv(url, header=None)                                                 # dataframe created


# Question 3.2

column_names =["Sex", "Length", "Diameter", "Height", "Whole weight",\
    "Shucked weight", "Viscera weight", "Shell weight", "Rings"] # column names as given
abalone.columns=column_names                 # naming the columns of dataframe with new column names
abalone=abalone.drop(columns=["Sex"])       # removing an entire column of " sex " as it's not required


# Question 3.3

x=(abalone.drop(columns=['Rings'])).values          # dataframe with no " Rings " column in it
y=abalone["Rings"].values                           # " rings elements stored with values"
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=333)           # from assignment file


# Question 3.4 

# Part 1

new_data_point = np.array([0.569552,0.446407,0.154437,1.016849,0.439051,0.222526,0.291208]) # new data point as given in question
# Computing the Euclidean distance between each row of x_train and new_data_point( given in assignment file)
distances = np.linalg.norm(x_train - new_data_point, axis=1) 

# Part 2

nearest_indices = distances.argsort()[:3]   # Get the indices of the 3 nearest neighbors

# Part 3

nearest_ages = y_train[nearest_indices]      # getting the ages of the nearest neighbors
age_mode = stats.mode(nearest_ages)     # find mode
print(" Predicted age :",age_mode," (Q3.4 [iii])")       # reporting mode of ages

# Part 4

predicted_ages_test = []

for data_point in x_test:   # For each row in X_test, perform prediction
    # Computing the Euclidean distance between each row of x_train and data_point( given in assignment file)
    distance = np.linalg.norm(x_train - data_point, axis=1)
    nearest_indices = distance.argsort()[:3]        # Get the indices of the 3 nearest neighbors
    nearest_ages = y_train[nearest_indices]     # getting the ages of the nearest neighbors
    mode_age = mode_calc(nearest_ages) # Finding the mode of age
    predicted_ages_test.append(mode_age)        # Adding to the list
    del distance,nearest_ages,nearest_indices

predicted_ages_test = np.array(predicted_ages_test)     # converting list to array
mse = mse_calculator(predicted_ages_test, y_test)        # Compute mean squared error (MSE)
print(" Mean Squared Error (MSE) :", mse," (Q3.4 [iv])")     # Report the mean squared error
del predicted_ages_test


# Question 3.5
 
mse_values=[]   # to store MSE value for each K(from 1 to 50)  neighbors
k_values=[]     # to store k values(from 1 to 50)

for k in range(1,51,1):
    
    predicted_ages_test = []

    for data_point in x_test:   # For each row in X_test, perform prediction
        # Computing the Euclidean distance between each row of x_train and data_point( given in assignment file)
        distance = np.linalg.norm(x_train - data_point, axis=1) 
        nearest_indices = distance.argsort()[:k]        # Get the indices of the 3 nearest neighbors
        nearest_ages = y_train[nearest_indices]     # getting the ages of the nearest neighbors
        mode_age = mode_calc(nearest_ages) # Finding the mode of age
        predicted_ages_test.append(mode_age)        # Adding to the list
        del distance,nearest_ages,nearest_indices

    predicted_ages_test = np.array(predicted_ages_test)     # converting list to array

    mse = mse_calculator(predicted_ages_test, y_test)        # Compute mean squared error (MSE)
    mse_values.append(mse)      # adding mse values to the list
    k_values.append(k)  # adding k values to the list

# converting lists to arrays
mse_values=np.array(mse_values)
k_values=np.array(k_values)

# The optimal K value
optimal_k = np.argmin(mse_values) + 1    # index of minimum element of MSE values to find it's corresponding K value
print(" Optimal K value :", optimal_k,"\n","Minimum MSE at optimal K is : ",mse_values[optimal_k-1])     # since indexing starts from 0, optimal K is (index +1)

# plotting the points
labels=np.arange(0,51,5)
plt.plot(k_values, mse_values, "b-o", label="MSE-Values")
plt.xticks(labels)
plt.xlabel("K ~ number of nearest neighbors")
plt.ylabel("Mean Squared Errors")
plt.title("MSE vs K neighbors")
plt.legend()
plt.grid(True)
#plt.savefig("MSE vs K neighbors.png")
plt.show()
