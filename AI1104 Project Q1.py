# Importing modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


def sigmoid(x):                 # the sigmoid function to return sigma(x) when x is the intput
    myValue= 1 / (1 + np.exp(-x))
    return myValue

def grad_sigmoid(x):            # the grad_sigmoid function to return derivative of sigma(x) when x is the intput
    myValue= np.exp(-x) / ((1 + np.exp(-x))**2)
    return myValue


# Question 1.1  [Dataset Preparation]

data=pd.read_csv("data_Q1.csv")     # loading the dataset
# converting each column of the dataframe to a numpy array and store them in arrays
training_x1, training_x2, training_y = data["x1"].to_numpy(),data["x2"].to_numpy(),data["y"].to_numpy()
N=len(training_y)       #length of each array
training_x = np.array([[x1,x2,1] for x1,x2
                       in zip(training_x1,training_x2)])        # training data array created from code in assignment


# Question 1.2  [Neural Network Preliminaries]

''' refer to assignment file inorder to get information of \
    "Neural Network Preliminaries" i.e terms , definations,\
    formulae and brief model of our Neural Network'''


# Question 1.3  [Definitions and Initialisations]

np.random.seed(123)  # Seed given in question for reproducibility

# Initialize weights matrices with random floating point numbers lying between [−1, 1]
weights1 = 2 * np.random.rand(3, 3) - 1     # W_1 -> given as weight matrix 1 with size 3 x 3
weights2 = 2 * np.random.rand(4, 1) - 1     # W_2 -> given as weight matrix 2 with size 4 x 1

learning_rate_gamma = 0.05  # Learning rate (γ), a hyperparameter
max_epochs = 100    # Maximum epochs
training_error = []     # Empty array to store training errors


# Question 1.4 [Training the Neural Network]

# Training loop
for epoch in range(max_epochs):
    
    
    # 1. Forward pass
    X = training_x      # step (a)
    H = np.dot(X, weights1)  # step (b) matrix multiplication of X & W_1
    Z = sigmoid(H)      # step (c)
    # step (d)
    bias_column = np.ones((Z.shape[0], 1))  # biased column
    Z = np.hstack((Z, bias_column))         # Append bias column to Z
    
    O = np.dot(Z, weights2)  # step (e)  matrix multiplication of Z & W_2
    y_hat = sigmoid(O)  # step (f)
    
    # step (g) Compute loss and append it to training_error array
    loss = np.mean((training_y - y_hat) ** 2)/2
    training_error.append(loss)
    
    
    # 2. Back propagation (formulae taken from assignment file)
    gradient2 = (np.dot(Z.T, (-(training_y.reshape(200,1) - y_hat) * grad_sigmoid(O))))/N
    temp = -(training_y.reshape(200,1) - y_hat) * grad_sigmoid(O) * grad_sigmoid(H)
    temp = temp @ np.diag(weights2[:-1].reshape(-1))
    gradient1 = (np.dot(X.T, temp))/N

    # Gradient descent
    weights2 -= learning_rate_gamma * gradient2
    weights1 -= learning_rate_gamma * gradient1


# Question 1.5 [Plotting training loss]

epochs=range(1, max_epochs + 1)
plt.plot(epochs, training_error,"b-",)
plt.xticks(range(0,101,10))
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Epochs')
plt.savefig('Training Loss vs Epochs.png')
plt.grid(True)
plt.show()
