#Ex 5 ML Coursera
library(tidyverse)
library(R.matlab)
source('linearRegCostFunction.R')
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
linearRegCostFunction_J(Xn, y, theta, 1)
linearRegCostFunction_G(Xn, y, theta, 1)
