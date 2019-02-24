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
theta<- c(0,0,0)

sigmoid <- function(z){
  g<- 1/(1+exp(1)^(-z))
  return(g)
}

costFunction_lg<- function(theta, X, y){
  eps<- 1e-5
  m <- length(y)
  h <- sigmoid(X%*%theta)
  l <- (1/m)*((-t(y)%*%log(h))-(t(1-y)%*%log(1-h+eps)))  
  return(l)
}

costFunction_lg(theta, X, y)



#A vectorized implementation is:
  #θ:=θ−m
  #αXT(g(Xθ)−y)

gradientDescent<-function(X, y, theta, alpha, num_iter){
  m <- length(y)
  J_hist <- rep(0, num_iter)
  for(i in 1:num_iter){
    h <- sigmoid(X%*%theta)
    theta<- theta-(alpha/m)*(t(X)%*%(h-y))
    J_hist[i]  <- costFunction_lg(theta,X, y)
  }
  results<-list(theta, J_hist)
  return(results)
}


#Learned: without the advance optimization algorithm, this takes a lot of time in my computer
theta<- c(0,0,0)
alpha <- .01
i <- 100000

results <- gradientDescent(X, y, theta, alpha, i)
theta <- c(results[[1]])
cost_hist <- results[[2]]
print(theta)
plot(1:i, cost_hist)

#############Predicting#########

theta2<-c(-25.161272,0.206233,0.201470) ##From octave's solution, as reference

##De-normalization of X


Xn<-data[,1:2]
Xn <- cbind(Xn, as.vector(matrix(1,1, length(data$exam1))))
Xn <- as.matrix(cbind(Xn[,3],Xn[,1],Xn[,2]))

prediction<-round(sigmoid(Xn%*%theta2))
1-sum(abs(prediction-y))/100  #89% w/ Octave results



prediction<-round(sigmoid(Xn%*%theta))
1-sum(abs(prediction-y))/100 # 60% with R's result (without advance optimization)

## Decision boundary at x2 = (1-/theta2)*(theta_0+theta1*x1)


###to do: calculate boundary
#theta2 (from octave)
m1<- (-theta2[1]/theta2[3])
b1<- (-theta2[2]/theta2[3])


m2<- (-theta[1]/theta[3])
b2<- (-theta[2]/theta[3])



data%>%ggplot()+geom_point(aes(x = exam1, y = exam2, color = accepted))+
  geom_abline(intercept = m1, slope = b1, color = "green")

