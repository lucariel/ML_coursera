costFunction<- function(X, y, theta){
  m <- nrow(X)
  predictions <- X%*%theta
  sqrErrors <- (predictions-y)^2
  J <- 1/(2*m)*sum(sqrErrors)
  J
}

X <- rbind(c(1,1), c(1,2), c(1,3))
class(X)

y <- as.vector(rbind(1,2,3))
theta<-as.vector(rbind(0,1))
costFunction(X, y, theta)
nrow(X)
X%*%theta
