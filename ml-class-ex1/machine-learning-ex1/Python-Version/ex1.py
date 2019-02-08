#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

path = os.getcwd()+"/Documents/ML Coursera/ml-class-ex1/ex1/Python-Version/"
os.getcwd()

#data = open(os.listdir(os.getcwd())[0]).read()
column_names = ['population','profit']
data = pd.read_csv('ex1data1.txt', sep=",", header=None, names = column_names)
#data.iloc[0:1,0:3]
#data.plot.scatter(x='population',
#                  y='profit',
#                  c='red')

#Add a column of ones to x
data['x0'] = 1
X = data.iloc[:,[2,0]]
y = np.array(data.iloc[:,1])
theta = np.array([0,0])
iterations = 1500
alpha = 0.01

def costFunction(X, y, theta):
    m = len(X)
    predictions = np.dot(X,theta)
    sqrErrors = np.array((predictions-y))**2
    J = 1/(2*m)*np.sum(sqrErrors)
    return J;


#def gradientDescent(X, y, theta, alpha, num_iter):
#    m= len(X)
#    for i in range(0, num_iter):
#        temp0 = 


# In[46]:


def gradientDescent(X, y, theta, alpha, num_iter):
    m= len(X)
    for i in range(0, num_iter):
        temp0 = theta[0]-(alpha/m)*sum((np.dot(X,theta)-y)*X.iloc[:,0])
        temp1 = theta[1]-(alpha/m)*sum((np.dot(X,theta)-y)*X.iloc[:,1])
        theta = np.array([temp0,temp1])
        costFunction(X, y, theta)
    print(costFunction(X, y, theta))
    


# In[47]:


gradientDescent(X,y,theta,0.01, 100)

