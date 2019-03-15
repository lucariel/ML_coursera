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
data_plot<-as.tibble(X)%>%ggplot(aes(V1, y))+
  geom_point(shape = 4, color = "red")+
  xlab("Change in water level (x)")+
  ylab("Water flowing out of the damn(y)")
  
##Init theta
theta <- c(1,1)
##Add x0 to X
Xn <- cbind( as.vector(matrix(1,1, length(X))),X)
Xnval<- cbind( as.vector(matrix(1,1, length(Xval))),Xval)
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
error_train<-learningCurve(Xn, y, Xnval, yval, 0)[[1]][-1]
error_val<-learningCurve(Xn, y, Xnval, yval, 0)[[2]][-1]
errorplot<-as.tibble(cbind(error_train,error_val, "m"=seq(1:length(error_val))))
errorplot%>%ggplot()+
  geom_line(aes(x= m, y = error_val, color = "Cross Validation"))+
  geom_line(aes(x= m, y = error_train, color = "Train"))+
  xlab("Number of training examples")+
  ylab("Error")+ggtitle("Learning Curve for linear regression")+
  scale_colour_manual("", 
                      breaks = c("Cross Validation", "Train"),
                      values = c("green", "blue"))
  
 #%% =========== Part 6: Feature Mapping for Polynomial Regression =============

p<-8


####For total X####
X_poly <- polyFeatures(X, p)
#View(X_poly)
X_poly <-(scale(X_poly))
#X_poly <- cbind( as.vector(matrix(1,1, dim(X_poly)[1])),X_poly)
####For Test set ########
X_poly_test<-polyFeatures(Xtest,p)
X_poly_test <- scale(X_poly_test)
#X_poly_test <- cbind( as.vector(matrix(1,1, dim(X_poly_test)[1])),X_poly_test)
####For Cross Validation set ########
Xval_poly<- polyFeatures(Xval, p)
Xval_poly<- scale(Xval_poly)
#Xval_poly <- cbind( as.vector(matrix(1,1, dim(Xval_poly)[1])),Xval_poly)
#=========== Part 7: Learning Curve for Polynomial Regression =============
theta_poly<-trainLinearReg(X_poly, y, 1)

X_polyn <- polyFeatures(X, p)
X_polyn <- cbind( as.vector(matrix(1,1, dim(X_polyn)[1])),X_polyn)
y_hat<-rowSums(X_polyn*theta_poly)
data_plot+geom_smooth(aes(y = y_hat))

################Polynomial Regression Learning Curve################

error_train<-learningCurve(X_poly, y, Xval_poly, yval, 2)[[1]][-1]
error_val<-learningCurve(X_poly, y, Xval_poly, yval, 1)[[2]][-1]


errorplot<-as.tibble(cbind(error_train,error_val, "m"=seq(1:length(error_val))))
errorplot%>%ggplot()+
  geom_line(aes(x= m, y = error_val, color = "Cross Validation"))+
  geom_line(aes(x= m, y = error_train, color = "Train"))+
  xlab("Number of training examples")+
  ylab("Error")+ggtitle("Learning Curve for polynomial regression")+
  scale_colour_manual("", 
                      breaks = c("Cross Validation", "Train"),
                      values = c("green", "blue"))


# =========== Part 8: Validation for Selecting Lambda =============
X_poly <- polyFeatures(X, p)
X_poly <-(scale(X_poly))
X_poly <- cbind( as.vector(matrix(1,1, dim(X_poly)[1])),X_poly)
Xval_poly<- polyFeatures(Xval, p)
Xval_poly<- scale(Xval_poly)
Xval_poly <- cbind( as.vector(matrix(1,1, dim(Xval_poly)[1])),Xval_poly)


sel_lambda<-validationCurve(X_poly, y,Xval_poly,yval)
lambdas<-sel_lambda[[1]]
error_train_l<-sel_lambda[[2]]
error_val_l<-sel_lambda[[3]]

plot_sel_lambda_data<-as.tibble(cbind(lambdas,error_train_l,error_val_l))
plot_sel_lambda_data%>%ggplot()+
  geom_line(aes(x= lambdas, y = error_val_l, color = "Cross Validation"))+
  geom_line(aes(x= lambdas, y = error_train_l, color = "Train"))+
  xlab("Number of training examples")+
  ylab("Error")+ggtitle("Lambda Selection")+
  scale_colour_manual("", 
                      breaks = c("Cross Validation", "Train"),
                      values = c("green", "blue"))

