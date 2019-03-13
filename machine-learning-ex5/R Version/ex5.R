#Ex 5 ML Coursera
library(tidyverse)
library(R.matlab)
source('linearReg.R')
source("lbfgsb3_.R")
#=========== Part 1: Loading and Visualizing Data =============
data<-readMat("ex5data1.mat")
##Training data
X<-data$X
y<-data$y

##Cross Validation data
Xval<-data$Xval
yval<-data$yval

##Test Data
Xtest<-data$Xtest
ytest<-data$ytest

###Ploting the data
as.tibble(X)%>%ggplot(aes(V1, y = y))+
  geom_point(shape = 4, color = "red")+
  xlab("Change in water level (x)")+
  ylab("Water flowing out of the damn(y)")
  
##Init theta
theta <- c(1,1)
##Add x0 to X
Xn <- cbind( as.vector(matrix(1,1, length(X))),X)
#=========== Part 2: Regularized Linear Regression Cost =============
linearRegCostFunction_J(Xn, y, 1)(theta)
#=========== Part 3: Regularized Linear Regression Gradient =============
linearRegCostFunction_G(Xn, y, 1)(theta)
#=========== Part 4: Train Linear Regression =============
thetas_lg<-trainLinearReg(Xn, y, 1)
linearRegCostFunction_J(Xn, y, 1)(thetas_lg)


###Plot fit over the data 

as.tibble(X)%>%ggplot(aes(V1, y = y))+
  geom_point(shape = 4, color = "red")+
  xlab("Change in water level (x)")+
  ylab("Water flowing out of the damn(y)")+
  geom_abline(slope =thetas_lg[2] ,intercept=thetas_lg[1], color="blue")
##=========== Part 5: Learning Curve for Linear Regression =============

