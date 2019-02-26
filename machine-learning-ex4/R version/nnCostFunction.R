nnCostFunction <- function(Theta1, Theta2, 
                           input_layer_size, 
                           hidden_layer_size, 
                           num_labels, 
                           X, y){
  source("aux_functions.R")
  
  #Theta1_unrolled<-nn_params[1:(input_layer_size+1)*hidden_layer_size]
  #Theta2_unrolled<-nn_params[(length(Theta1_unrolled)+1):length(nn_params)]
  
  #sTheta1<-roll_into_matrix(Theta1_unrolled, hidden_layer_size, hidden_layer_size*(input_layer_size+1))
  #Theta2<-roll_into_matrix(Theta2_unrolled,  num_labels, num_labels*(hidden_layer_size+1))
  
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
  Y<- matrix(0, nrow = m, ncol = 10)
  for(i in 1:m){
    Y[i,y[i]] = 1
  }
  ## Cost Calculation
  eps<- 1e-5 ## To avoid log(0) == -Inf
  J <- sum(colSums(-Y*log(h)-(1-Y)*log(1-h+eps)))/m
  J
}
