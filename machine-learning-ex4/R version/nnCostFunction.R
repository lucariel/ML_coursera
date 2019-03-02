nnCostFunction <- function(input_layer_size, 
                           hidden_layer_size, 
                           num_labels, 
                           X, y, lambda){
  source("aux_functions.R")

  function(nn_params) {
  ##Roll Thetas to get back the matrix
  Theta1_unrolled_f<-nn_params[1:((input_layer_size+1)*hidden_layer_size)]
  Theta2_unrolled_f<-nn_params[(length(Theta1_unrolled)+1):length(nn_params)]
  
  Theta1<-roll_into_matrix(Theta1_unrolled_f, hidden_layer_size, hidden_layer_size*(input_layer_size+1))
  Theta2<-roll_into_matrix(Theta2_unrolled_f,  num_labels, num_labels*(hidden_layer_size+1))
  
  #### Setting up useful variables
  m <- dim(X)[1]
  
  #### Return variables 
  J <- 0
  Theta1_grad <- matrix(0, nrow = dim(Theta1)[1], ncol = dim(Theta1)[2])
  Theta2_grad <- matrix(0, nrow = dim(Theta2)[1], ncol = dim(Theta2)[2])
  
  
  ########## Forward propagation
  a1 <-cbind(matrix(1, nrow = m, ncol = 1), X)
  z2 <- a1%*%t(Theta1)
  a2 <- sigmoid(z2)
  a2 <-cbind(matrix(1, nrow = dim(a2)[1], ncol = 1), a2)
  z3 <- a2%*%t(Theta2)
  a3 <- sigmoid(z3)
    
  h <- a3
  
  ## y transformation into Y so that each row is an example
  Y<- matrix(0, nrow = m, ncol = num_labels)
  for(i in 1:m){
    Y[i,y[i]] = 1
  }
  ##Regularization term
  
  #regularization = *(sum(sumsq(Theta1(:,2:end)))+sum(sumsq(Theta2(:,2:end))))
  theta_1sq<-Theta1[1:dim(Theta1)[1],2:dim(Theta1)[2]]*Theta1[1:dim(Theta1)[1],2:dim(Theta1)[2]]
  theta_2sq<-Theta2[1:dim(Theta2)[1],2:dim(Theta2)[2]]*Theta2[1:dim(Theta2)[1],2:dim(Theta2)[2]]
  regularization = (lambda/(2*m))*(sum(colSums(theta_1sq))+sum(colSums(theta_2sq)))
  
  
  ## Cost Calculation
  eps<- 1e-5 ## To avoid log(0) == -Inf
  J <- sum(colSums(-Y*log(h)-(1-Y)*log(1-h+eps)))/m+regularization
  J
  
  ##
  for(t in 1:m){
    ########FORWARD PROPAGATION################
    a1<-X[t,]
    a1<-c(1,a1)
    
    z2<-Theta1%*%a1
    a2<-sigmoid(z2)
    a2<-c(1,a2)
    
    z3<-Theta2%*%a2
    a3<-sigmoid(z3)
    
    ######Deltas#########
    delta3<-(a3-(Y[t,]))
    delta2<-(t(Theta2)%*%delta3)*(a2*(1-a2))
    delta2<-delta2[2:length(delta2)]
    
    Theta1_grad <-  Theta1_grad + delta2%*%t(a1)
    Theta2_grad <-  Theta2_grad + delta3%*%t(a2)
  }
  
  #+ (lambda/m)*[zeros(size(Theta2, 1), 1), Theta2(:,2:end)];
  cbind(0,Theta2)
  Theta1_grad<-Theta1_grad/m+(lambda/m)*(cbind(0,Theta1[1:dim(Theta1)[1],2:dim(Theta1)[2]]))
  Theta2_grad<-Theta2_grad/m+(lambda/m)*(cbind(0,Theta2[1:dim(Theta2)[1],2:dim(Theta2)[2]]))
  
  ##Unroll gradients
  Theta1_grad<-matrix(Theta1_grad, ncol = 1, byrow = F)
  Theta2_grad<-matrix(Theta2_grad, ncol = 1, byrow = F)
  
  list(J, list(Theta1_grad,Theta2_grad))
  return(J)}
}

nnGradFunction <- function(input_layer_size, 
                           hidden_layer_size, 
                           num_labels, 
                           X, y, lambda){
  function(nn_params) {
  source("aux_functions.R")
  
  
  ##Roll Thetas to get back the matrix
  Theta1_unrolled_f<-nn_params[1:((input_layer_size+1)*hidden_layer_size)]
  Theta2_unrolled_f<-nn_params[(length(Theta1_unrolled)+1):length(nn_params)]
  
  Theta1<-roll_into_matrix(Theta1_unrolled_f, hidden_layer_size, hidden_layer_size*(input_layer_size+1))
  Theta2<-roll_into_matrix(Theta2_unrolled_f,  num_labels, num_labels*(hidden_layer_size+1))
  
  #### Setting up useful variables
  m <- dim(X)[1]
  
  #### Return variables 
  J <- 0
  Theta1_grad <- matrix(0, nrow = dim(Theta1)[1], ncol = dim(Theta1)[2])
  Theta2_grad <- matrix(0, nrow = dim(Theta2)[1], ncol = dim(Theta2)[2])
  
  
  ########## Forward propagation
  a1 <-cbind(matrix(1, nrow = m, ncol = 1), X)
  z2 <- a1%*%t(Theta1)
  a2 <- sigmoid(z2)
  a2 <-cbind(matrix(1, nrow = dim(a2)[1], ncol = 1), a2)
  z3 <- a2%*%t(Theta2)
  a3 <- sigmoid(z3)
  
  h <- a3
  
  ## y transformation into Y so that each row is an example
  Y<- matrix(0, nrow = m, ncol = num_labels)
  for(i in 1:m){
    Y[i,y[i]] = 1
  }
  ##Regularization term
  
  #regularization = *(sum(sumsq(Theta1(:,2:end)))+sum(sumsq(Theta2(:,2:end))))
  theta_1sq<-Theta1[1:dim(Theta1)[1],2:dim(Theta1)[2]]*Theta1[1:dim(Theta1)[1],2:dim(Theta1)[2]]
  theta_2sq<-Theta2[1:dim(Theta2)[1],2:dim(Theta2)[2]]*Theta2[1:dim(Theta2)[1],2:dim(Theta2)[2]]
  regularization = (lambda/(2*m))*(sum(colSums(theta_1sq))+sum(colSums(theta_2sq)))
  
  
  ## Cost Calculation
  eps<- 1e-5 ## To avoid log(0) == -Inf
  J <- sum(colSums(-Y*log(h)-(1-Y)*log(1-h+eps)))/m+regularization
  J
  
  ##
  for(t in 1:m){
    ########FORWARD PROPAGATION################
    a1<-X[t,]
    a1<-c(1,a1)
    
    z2<-Theta1%*%a1
    a2<-sigmoid(z2)
    a2<-c(1,a2)
    
    z3<-Theta2%*%a2
    a3<-sigmoid(z3)
    
    ######Deltas#########
    delta3<-(a3-(Y[t,]))
    delta2<-(t(Theta2)%*%delta3)*(a2*(1-a2))
    delta2<-delta2[2:length(delta2)]
    
    Theta1_grad <-  Theta1_grad + delta2%*%t(a1)
    Theta2_grad <-  Theta2_grad + delta3%*%t(a2)
  }
  
  #+ (lambda/m)*[zeros(size(Theta2, 1), 1), Theta2(:,2:end)];
  cbind(0,Theta2)
  Theta1_grad<-Theta1_grad/m+(lambda/m)*(cbind(0,Theta1[1:dim(Theta1)[1],2:dim(Theta1)[2]]))
  Theta2_grad<-Theta2_grad/m+(lambda/m)*(cbind(0,Theta2[1:dim(Theta2)[1],2:dim(Theta2)[2]]))
  
  ##Unroll gradients
  Theta1_grad<-matrix(Theta1_grad, ncol = 1, byrow = F)
  Theta2_grad<-matrix(Theta2_grad, ncol = 1, byrow = F)
  
  grad <- c(as.vector(Theta1_grad), as.vector(Theta2_grad))
  return(grad)
  }
  }
