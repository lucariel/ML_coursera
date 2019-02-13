#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Logistic Regression Python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# In[15]:


##Preparing Data (1)
path = os.getcwd()+"/Documents/ML Coursera/ml-class-ex1/ex1/Python-Version/"
column_names =  ['exam1','exam2','accepted']
data = pd.read_csv('ex2data1.txt', sep=",", header=None, names = column_names)
data['x0'] = 1


# In[20]:


##Preparing Data (1) {Initialization of variables}
X = data.iloc[:,[3,0,1]]
y = np.array(data.iloc[:,2])
theta = np.array([0,0,0])
iterations = 1500
alpha = 0.01


# In[33]:


def sigmoid(z):
    g = 1/(1+np.e**(-z))
    return g;


def costFunction(X, y, theta):
    m = len(X)
    h = sigmoid(np.dot(X,theta))
    l = (1/m)*((np.dot(-np.transpose(y),np.log(h)))-(np.dot(np.transpose(1-y),np.log(1-h))))
    return l;

costFunction(X, y, theta)#For initial parameter <- 0.6931471805599452 (OK)


# In[37]:


def gradientDescent(X, y, theta, alpha, num_iter):
    m = len(y)
    for i in range(0, num_iter):
        h = sigmoid(np.dot(X,theta))
        theta = theta - (alpha/m)*np.dot(np.transpose(X), (h-y))
    return(theta)
    

gradientDescent(X,y,theta, 0.005, 500000)

