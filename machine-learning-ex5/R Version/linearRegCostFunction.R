#linearRegCostFunction#

linearRegCostFunction_J<-function(X, y, theta, lambda){
  m <- nrow(X)
  sqrErrors <- ((X%*%theta)-y)^2
  J <- 1/(2*m)*sum(sqrErrors)
  theta_reg <- theta
  theta_reg<-sum(theta_reg^2)
  theta_reg[1]<-0
  reg<-lambda/(2*m)*theta_reg
  J
}

linearRegCostFunction_G<-function(X, y, theta, lambda){
  m<-length(y)
  theta_reg <- theta
  theta_reg[1]<-0
  grad <- (1/m)*t((X%*%theta)-y)%*%X+(lambda/m)*theta_reg
  grad
}

