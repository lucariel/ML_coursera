function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 

%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 

parameters_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

sel = []
for i = 1:(size(parameters_values)(2))
  for j = 1:(size(parameters_values)(2))
    C = parameters_values(i);
    sigma = parameters_values(j);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

    predictions = svmPredict(model, Xval);
    sel = [sel ; C sigma mean(double(predictions ~= yval))];
  endfor
endfor
errors = sel(:,3);
[minval, row] = min(min(errors,[],2));

%

C = sel(row,1)
sigma = sel(row,2)





% =========================================================================

end
