##### Neural Networks - Ex 3, Ex 4 Coursera
library(tidyverse)
library(R.matlab)
source("checkNNGradients.R")
###Loading data
data<-readMat("ex4data1.mat")

X <- data$X
y <- as.vector(data$y)
m <- dim(X)[1]
###Loading Thetas
Thetas<-readMat("ex4weights.mat")
##Asigning for better work
Theta1<-Thetas$Theta1
Theta2<-Thetas$Theta2



input_layer_size  <- 400 # 20x20 Input Images of Digits
hidden_layer_size <- 25  # 25 hidden units
num_labels <- 10         # 10 labels, from 1 to 10   

########################Vizualization of Data#################################
# get 100 random rows of X
Xn <- X[sample(nrow(X), size=100, replace=FALSE),]
#View(X)
# allocate empty image matrix (200 by 200 pixels)
Z <- matrix(rep(0, length(Xn)), nrow=200)

# fill empty image matrix
for (row in 0:9) {
  rmin <- 1 + (row)*20
  for (col in 0:9) {
    cmin <- 1 + (col)*20
    Z[rmin:(rmin+19), cmin:(cmin+19)] <- Xn[row * 10 + col + 1,]
  }
}
# plot (after rotating matrix 90 degrees)
image(t(apply(Z, 2, rev)))
##################################End Vizualizatio##########################

###Unrolling 
Theta1_unrolled<-matrix(Theta1, ncol = 1, byrow = F)
Theta2_unrolled<-matrix(Theta2, ncol = 1, byrow = F)

nn_params<-rbind(Theta1_unrolled,Theta2_unrolled)


##Cost Calculation 
source("nnCostFunction.R")
source("aux_functions.R")

##Cost with lambda = 1
J<-nnCostFunction(input_layer_size, hidden_layer_size, num_labels, 
X, y, 1)
J(nn_params)#0.3836783

## ================ Initializing Pameters ================
initial_Theta1<- randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2<- randInitializeWeights(hidden_layer_size, num_labels)

initial_Theta1_unrolled<-matrix(initial_Theta1, ncol = 1, byrow = F)
initial_Theta2_unrolled<-matrix(initial_Theta2, ncol = 1, byrow = F)

initial_nn_params<-c(initial_Theta1_unrolled,initial_Theta2_unrolled)

## ================ Training NN  ================

lambda <- 1
##Creating ShortCuts
costFunction <- nnCostFunction(input_layer_size, hidden_layer_size, 
                               num_labels, X, y, lambda) 

gradFunction <- nnGradFunction(input_layer_size, hidden_layer_size, 
                               num_labels, X, y, lambda) 

source("lbfgsb3_.R") #simil fmincg
opt <- lbfgsb3_(initial_nn_params, fn = costFunction, gr=gradFunction,
                control = list(trace=1,maxit=50))


## Get trained parameters
nn_params <- opt$prm
cost <- opt$f

#  Theta1 and Theta2 trained
Theta1 <- matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))],
                 hidden_layer_size, (input_layer_size + 1))

Theta2 <- matrix(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):length(nn_params)],
                 num_labels, (hidden_layer_size + 1))

pred <- predict(Theta1, Theta2, X)
# accuracy
print(mean(pred==y))
