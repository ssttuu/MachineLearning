function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Cost Function
hypothesis = sigmoid( X * theta);

tmpTheta = theta;
tmpTheta(1) = 0;

costP = (-1 .* y)' * log( hypothesis);
costN = (1 .- y)' * log( 1 - hypothesis);

jReg = (lambda/ (2*m)) * (tmpTheta' * tmpTheta);

J = (1/m) * (costP - costN) + jReg;


gReg = (lambda/ m) * tmpTheta;

grad = (1/m) * (X' * (hypothesis - y)) + gReg;

% =============================================================

end;
