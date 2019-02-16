##Logistic regression
library(tidyverse)
library(ggplot2)

e<-exp(1)

data <- read.delim("ex2data1.txt", header = F, sep = ",")
colnames(data)<-c("exam1", "exam2", "accepted")
data%>%ggplot(aes(x = exam1, y = exam2, color = accepted))+geom_point()
X<-data[,1:2]
X <- cbind(X, as.vector(matrix(1,1, length(data$exam1))))
X <- as.matrix(cbind(X[,3],X[,1],X[,2]))

y<-data[,3]
theta<- as.vector(c(0,0,0))

sigmoid <- function(z){
  g<- 1/(1+exp(1)^(-z))
  return(g)
}

costFunction_lg<- function(theta, X, y){
  m <- length(y)
  h <- sigmoid(X%*%theta)
  l <- (1/m)*((-t(y)%*%log(h))-(t(1-y)%*%log(1-h)))  
  return(l)
}

costFunction_lg(theta, X, y)



#A vectorized implementation is:
  #θ:=θ−m
  #αXT(g(Xθ)−y)

gradientDescent<-function(X, y, theta, alpha, num_iter){
  control<- costFunction_lg(theta, X, y)
  m <- length(y)
  for(i in 1:num_iter){
    h <- sigmoid(X%*%theta)
    theta<- theta-(alpha/m)*t(X)%*%(h-y)
    control<-rbind(control,costFunction_lg(theta, X, y))
  }
  return(theta)

}

gradientDescent(X,y,theta, 0.005, 10)


