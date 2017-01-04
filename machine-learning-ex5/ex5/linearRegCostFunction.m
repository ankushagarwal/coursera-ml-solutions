function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
% X     12x2
% y     12x1
% theta 2x1
% disp(size(X));
% disp(size(y));
% disp(size(theta));
t = (X * theta) - y;
t = t .* t;
J = (1/(2*m))*sum(t);
theta1 = theta(2:size(theta, 1), 1);
J = J + (lambda/(2*m))*sum(theta1 .* theta1);

% 2x12 12x1
grad = (X' * ((X * theta) - y))/m + (lambda/m)*theta;
grad(1, 1) = grad(1, 1) - (lambda/m)*theta(1, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
