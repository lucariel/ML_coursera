#!/usr/bin/env python
# coding: utf-8

# In[79]:


#Logistic Regression Python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from sklearn import svm


# In[37]:


##Preparing Data (1)
path = os.getcwd()+"/Documents/ML Coursera/ml-class-ex1/ex1/Python-Version/"
column_names =  ['exam1','exam2','accepted']
data = pd.read_csv('ex2data1.txt', sep=",", header=None, names = column_names)
data_plt = data


# In[38]:


##Preparing Data (1) {Initialization of variables}
data['x0'] = 1
X = data.iloc[:,[3,0,1]]
y = np.array(data.iloc[:,2])
theta = np.array([0,0,0])
iterations = 1500
alpha = 0.01


# In[39]:


##Plotting initial classes
import seaborn as sns
df = sns.load_dataset('iris')

data = data_plt
sns.scatterplot(data = data , x='exam1', y='exam2', hue = 'accepted')


# In[40]:


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


# In[41]:


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
    


# In[55]:


#Results
theta_result = gradientDescent(X,y,theta, 0.0005, 500000)["theta"];
J_hist = gradientDescent(X,y,theta, 0.0005, 500000)["J_hist"];



# In[49]:


J_hist
plt.plot(J_hist) ##Showing descending Cost over iterations


# In[65]:


##Prediction


#Control Theta from Octave - Without advance optimization, takes too long
theta_from_octave = np.array([-25.161272,0.206233,0.201470])

prediction=np.round(sigmoid(np.dot(X,theta_from_octave)))

1-np.sum(np.abs(prediction-y))/len(y)
#89% Acc


# In[150]:


##Calculate m and b for Decision Boundary 
m= (-theta_from_octave[0]/theta_from_octave[2])
b= (-theta_from_octave[1]/theta_from_octave[2])


# In[152]:


## Decision boundary
plt.scatter(data['exam1'], data['exam2'], c = data['accepted'])


# draw Decision boundary  line 
plt.plot([0, (-m/b)], [m, 0], 'k-')

plt.show()


# In[ ]:




