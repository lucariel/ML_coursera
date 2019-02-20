#!/usr/bin/env python
# coding: utf-8

# In[209]:


#Logistic Regression Python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math


# In[3]:


##Preparing Data (1)
path = os.getcwd()+"/Documents/ML Coursera/ml-class-ex1/ex1/Python-Version/"
column_names =  ['exam1','exam2','accepted']
data = pd.read_csv('ex2data1.txt', sep=",", header=None, names = column_names)
data['x0'] = 1


# In[35]:


##Preparing Data (1) {Initialization of variables}
X = data.iloc[:,[3,0,1]]
y = np.array(data.iloc[:,2])
theta = np.array([0,0,0])
iterations = 1500
alpha = 0.01


# In[252]:


##Plotting initial classes




# In[225]:


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def costFunction(X, y, theta):
    epsilon = 1e-5 #To avoid the log(0)-> -Inf
    m = len(X)
    h = sigmoid(np.dot(X,theta))      
    l = (1/m)*((np.dot(-np.transpose(y),np.log(h)))-(np.dot(np.transpose(1-y),np.log(1-h+epsilon))))
    return(l);

costFunction(X, y, theta)#For initial parameters <- 0.6931471805599452 (OK)


# In[245]:


def gradientDescent(X, y, theta, alpha, num_iter):
    m = len(y)
    J_hist = np.zeros(num_iter)
    for i in range(0, num_iter):   
        print(costFunction(X, y, theta))
        print(theta)
        h = sigmoid(np.dot(X,theta))
        theta = theta - (alpha/m)*np.dot(np.transpose(X), (h-y));
        J_hist[i] =  costFunction(X, y, theta);
        
        
        
    results = {
       "theta": theta,
       "J_hist": J_hist
    }
    return(results)
    


# In[246]:


theta_result = gradientDescent(X,y,theta, 0.0005, 100)["theta"];
J_hist = gradientDescent(X,y,theta, 0.0005, 100)["J_hist"];


# In[249]:


J_hist
plt.plot(J_hist) ##Showing descending Cost over iterations


# In[ ]:


##Results
gradientDescent(X,y,theta, 0.005, 500000)


# In[ ]:


##Make predictions 


# In[ ]:


## Decision boundary

