#ex1multi

##Ex 1 ML Coursera##
library(tidyverse)
library(ggplot2)

data <- read.delim("ex1data2.txt", header = F, sep = ",")
#data%>%ggplot(aes(x = V1, y = V2))+
#  geom_point(shape = 4, color = "red")+
#  labs(x = "Population of Cityin 10,000s", y = "Profit in $10,000s")+
  
  ## Getting proper X matrix
X = cbind(data, as.vector(matrix(1,1, length(data$V1))))
X = cbind(X[,4],X[,1], X[,2])
colnames(X)<- c("x0", "size", "rooms")
head(X,2)
#Initializating rest of variables
y = as.vector(data[,3])
theta<-as.vector(rbind(0,0,0))



##Normalization features
X[,2] <- (X[,2]-mean(X[,2]))/sd(X[,2])
y <- (y-mean(y))/sd(y)
X[,3] <- (X[,3]-mean(X[,3]))/sd(X[,3])

##Vectorized Cost Function
costFunction<- function(X, y, theta){
  m <- nrow(X)
  predictions <- X%*%theta
  sqrErrors <- (predictions-y)^2
  J <- 1/(2*m)*sum(sqrErrors)
  J
}
alpha <-0.01
##Vectorized gradientDescent
theta<-as.vector(rbind(0,0,0))

gradientDescent<-function(X, y, theta, alpha, num_iter){
  m <- nrow(X)
  for(i in 1:num_iter){
    #theta - (alpha/m*sum((X*theta - y).* X))';
    theta<-as.vector(theta - t(alpha/m*colSums(as.vector(X%*%theta-y)*X)))
    costFunction(X, y, theta)
    
  }
  theta
  costFunction(X, y, theta)
  }


gradientDescent(X,y,theta,0.01, 1500)

##Normal equation
#theta = pinv((X'*X))*X'*y
#install.packages("matlib")
theta<-as.vector(solve((t(X)%*%X))%*%t(X)%*%y)
costFunction(X, y, theta)

