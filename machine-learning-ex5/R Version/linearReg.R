#linearRegCostFunction#

linearRegCostFunction_J<-function(X, y, lambda){
  J_func<-function(theta){
    m <- nrow(X)
    sqrErrors <- ((X%*%theta)-y)^2
    J <- 1/(2*m)*sum(sqrErrors)
    theta_reg <- theta
    theta_reg<-sum(theta_reg^2)
    theta_reg[1]<-0
    reg<-lambda/(2*m)*theta_reg
    J
  }
  
}

linearRegCostFunction_G<-function(X, y,lambda){
  grad_func<-function(theta){
    m<-length(y)
    theta_reg <- theta
    theta_reg[1]<-0
    grad <- (1/m)*t((X%*%theta)-y)%*%X+(lambda/m)*theta_reg
    grad
  }
}

trainLinearReg<-function(X, y, lambda){
  incostFunction<-linearRegCostFunction_J(X, y, lambda)
  ingradFunction<-linearRegCostFunction_G(X, y, lambda)
  init_theta<-rep(0,dim(X)[2])
  opt<-optim(init_theta,incostFunction,ingradFunction )
  opt$par
}


learningCurve<-function(X, y, Xval, yval, lambda){
  m<-dim(X)[1]
  error_train<-rep(0,m)
  error_val<-rep(0,m)
  for (i in (2:m)){
    X_train <- X[1:i,]
    y_train <- y[1:i]
    theta_tr<-trainLinearReg(X_train,y_train,lambda)
    error_train[i]<-linearRegCostFunction_J(X_train,y_train,lambda)(theta_tr)
    error_val[i]<-linearRegCostFunction_J(Xval,yval,lambda)(theta_tr)
  }
  theta_tr
  errors<-list(error_train,error_val)
  errors
}


polyFeatures<-function(X,p){
  X_poly<-X
  for(i in(2:p)){
    X_poly<-cbind(X_poly, X_poly[,1]^i)
  }
  X_poly
}

