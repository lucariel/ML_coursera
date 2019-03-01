##Auxiliar Functions of ex4.R###


roll_into_matrix<-function(v, i, j){
  new_matrix <- v[1:i]
  for(k in seq(from = (i+1), to =j , by = i)){
    new_matrix <- cbind(new_matrix, v[(k): (k+i-1)])
    
  }
  new_matrix
}


sigmoid <- function(z){
  g<- 1/(1+exp(1)^(-z))
  return(g)
}


sigmoid_gradient<-function(z){
  sigmoid(z)*(1-sigmoid(z))
}


randInitializeWeights<- function(L_in, L_out){
  epsilon_init <- 0.12
  W<-replicate(L_in+1, runif(L_out, min = -epsilon_init, max = epsilon_init)) 
  W
}


debugInitializeWeights = function (fanOut, fanIn)
{
  w = matrix(sin(1:(fanOut*(1+fanIn))), fanOut, 1+fanIn)/10
  return(w)
}