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
##Normalize features

X[,2] <- (X[,2]-mean(X[,2]))/sd(X[,2])
X[,3] <- (X[,3]-mean(X[,3]))/sd(X[,3])



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
  J_hist <- rep(0, num_iter)
  m <- length(y)
  for(i in 1:num_iter){
    h <- sigmoid(X%*%theta)
    theta<- theta-(alpha/m)*t(X)%*%(h-y)
    J_hist[num_iter]<-costFunction_lg(theta, X, y)
  }
  return(theta)

}


#Learned: without the advance optimization algorithm, this takes a lot of time in my computer
alpha <- .1
i <- 200000
results <- gradDescent(X, y, theta, 0.001, 200000)
theta <- results[[1]]
cost_hist <- results[[2]]
print(theta)
plot(1:i, cost_hist)

#############Predicting#########

theta2<-c(-25.161272,0.206233,0.201470) ##From octave's solution, as reference

##De-normalization of X


Xn<-data[,1:2]
Xn <- cbind(X, as.vector(matrix(1,1, length(data$exam1))))
Xn <- as.matrix(cbind(X[,1],X[,2],X[,3]))


prediction<-round(sigmoid(Xn%*%theta))
1-sum(abs(prediction-y))/100 


## Decision boundary at x2 = (1-/theta2)*(theta_0+theta1*x1)

###to do: calculate boundary

data%>%ggplot(aes(x = exam1, y = exam2, color = accepted))+geom_point()+
  geom_segment(aes(x = 100, y = 0, xend = 0, yend = 100))

