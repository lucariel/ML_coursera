#!/usr/bin/env python
# coding: utf-8

# In[2]:


###ex1multi
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# In[28]:


path = os.getcwd()+"/Documents/ML Coursera/ml-class-ex1/ex1/Python-Version/"
os.getcwd()
column_names = ['size','rooms','price']

data = pd.read_csv('ex1data2.txt', sep=",", header=None, names = column_names)


# In[78]:


data['x0'] = 1
#print(data)
X = data.iloc[:,[3,0,1]]
#print(X)
y = np.array(data.iloc[:,2])
#print(y)
theta = np.array([0,0,0])
iterations = 1500
alpha = 0.01


# In[79]:


#Cost Function already for n variables
def costFunction(X, y, theta):
    m = len(X)
    predictions = np.dot(X,theta)
    sqrErrors = np.array((predictions-y))**2
    J = 1/(2*m)*np.sum(sqrErrors)
    return J;


# In[81]:


##Normalization of variables
X.iloc[:,1] = (X.iloc[:,1]-np.mean(X.iloc[:,1]))/np.std(X.iloc[:,1])
X.iloc[:,2] = (X.iloc[:,2]-np.mean(X.iloc[:,2]))/np.std(X.iloc[:,2])
y=(y-np.mean(y))/np.std(y)


# In[104]:


def gradientDescent(X, y, theta, alpha, num_iter):
    m= len(X)
    for i in range(0, num_iter):
        theta = np.array(theta-np.transpose(alpha/m*np.dot(np.array(np.dot(X,theta)-y),X)))
    return costFunction(X, y, theta)


# In[106]:


print(gradientDescent(X,y,theta,0.01, 1500))


# In[113]:


#Normal Eq
theta=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.transpose(X)),y)

