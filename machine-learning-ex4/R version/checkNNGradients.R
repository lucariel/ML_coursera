#########checkNNGradients and auxiliar functions for the checking
checkNNGradients<-function(lambda = 0){
  source("aux_functions.R")
  source("nnCostFunction.R")
  
  input_layer_size <- 3
  hidden_layer_size <- 5
  num_labels <- 3
  m <- 5
  
  # We generate some 'random' test data
  Theta1 <- debugInitializeWeights(hidden_layer_size, input_layer_size)
  Theta2 <- debugInitializeWeights(num_labels, hidden_layer_size)
  ##Reusing debugInitializeWeights to generate X
  X  <- debugInitializeWeights(m, (input_layer_size - 1))
  y  <- 1 + seq(1:m)%%num_labels
  #######Unroll parameters
  Theta1_unrolled<-matrix(Theta1, ncol = 1, byrow = F)
  Theta2_unrolled<-matrix(Theta2, ncol = 1, byrow = F)
  
  nn_params<-rbind(Theta1_unrolled,Theta2_unrolled)
  
  
  
  grad <- gradFunc(nn_params) 
  numgrad <- computeNumericalGradient(costFunc, nn_params)
  
  print(cbind(numgrad, grad))
  writeLines("The above two columns you get should be very similar")
  diff = norm(as.matrix(numgrad - grad),"f")/norm(as.matrix(numgrad + grad),"f")
  cat(sprintf(c("If your backpropagation implementation is correct, then\n",
                "the relative difference will be small. \n",
                "\nRelative Difference: %g\n"), diff))
}


computeNumericalGradient <- function (J, theta)
{
  numgrad <- rep(0, length(theta))
  perturb <- numgrad
  e <- 1e-4
  for (p in 1:length(theta))
  {
    perturb[p] <- e
    loss1 <-J(theta - perturb)
    loss2<- J(theta + perturb)
    numgrad[p] <- (loss2 - loss1)/(2*e)
    perturb[p] <- 0
  }
  return(numgrad)
}


costFunc <- function(p){
  ##Getting only Cost 
  results<-nnCostFunction(nn_params = p, 
                          input_layer_size, 
                          hidden_layer_size, 
                          num_labels, 
                          X, y, lambda)
  results<-(results[[1]])
  
  
}
gradFunc <- function(p){
  ##Getting only thetas_grad
  results<-nnCostFunction(nn_params = p, 
                          input_layer_size, 
                          hidden_layer_size, 
                          num_labels, 
                          X, y, lambda)
  results<-unlist(results[[2]])
  results
  
}

