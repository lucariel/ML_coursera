##Auxiliar Functions###


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