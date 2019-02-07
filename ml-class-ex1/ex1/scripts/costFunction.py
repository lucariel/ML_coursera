#!/usr/bin/env python
# coding: utf-8

# In[5]:


##Cost Function in Python
import numpy as np


# In[38]:


def costFunction(X, y, theta):
    m = len(X)
    predictions = np.dot(X,theta)
    sqrErrors = np.array((predictions-y))**2
    J = 1/(2*m)*np.sum(sqrErrors)
    return J;


X = np.matrix([[1, 1], [1, 2], [1,3]])
y = np.array([1,2,3])
theta = np.array([0,1])
costFunction(X, y, theta)


# In[23]:




