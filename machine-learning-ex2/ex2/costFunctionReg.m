function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
J = ((-1 / m) * ((y' * log(sigmoid(X * theta))) + ((1-y') * log(1 - sigmoid(X * theta))))) + ((lambda/(2*m))*(sum(power(theta, 2)) - power(theta(1), 2)));

grad1 = (((sigmoid(X * theta) - y)' * X) / m)';
grad = ((((sigmoid(X * theta) - y)' * X) / m)' + ((lambda / m) * theta));
grad(1) = grad1(1);
end
