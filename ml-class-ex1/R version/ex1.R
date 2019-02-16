##Ex 1 ML Coursera##
library(tidyverse)
library(ggplot2)

data <- read.delim("ex1data1.txt", header = F, sep = ",")
data%>%ggplot(aes(x = V1, y = V2))+
  geom_point(shape = 4, color = "red")+
  labs(x = "Population of Cityin 10,000s", y = "Profit in $10,000s")
  
## Getting proper X matrix
colnames(data)<-c("population", "profit")
X = cbind(data, as.vector(matrix(1,1, length(data$population))))
X = cbind(X[,3],X[,1])
colnames(X)<- c("x0", "population")
head(X,2)
#Initializating rest of variables
y = as.vector(data[,2])
theta<-as.vector(rbind(0,0))



costFunction<- function(X, y, theta){
  m <- nrow(X)
  predictions <- X%*%theta
  sqrErrors <- (predictions-y)^2
  J <- 1/(2*m)*sum(sqrErrors)
  J
}

costFunction(X, y, theta)

gradientDescent<-function(X, y, theta, alpha, num_iter){
  m <- nrow(X)
  for(i in 1:num_iter){
    temp0<-theta[1]-(alpha/m)*sum((X%*%theta-y)*X[,1])
    temp1<-theta[2]-(alpha/m)*sum((X%*%theta-y)*X[,2])
    theta<-c(temp0,temp1)
    costFunction(X, y, theta)
    
  }
  costFunction(X, y, theta)
  theta
}
theta<-as.vector(rbind(0,0))

gradientDescent(X,y,theta,0.01, 100)
