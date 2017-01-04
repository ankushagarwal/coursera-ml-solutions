function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
% X  5000 x 401
% T1 = 25 x 401
% 5000 x 26
% T2 = 10 x 26
% 10 x 5000
% y  = 5000 x 1
tempX = X;
X = [ones(size(X, 1), 1) X];
X = sigmoid(X * Theta1');
X = [ones(size(X, 1), 1) X];
p = (sigmoid(X * Theta2'))'; % 10x5000
y1 = zeros(num_labels, size(y, 1)); % 10x5000
for i = 1:size(y, 1)
  y1(y(i, 1), i) = 1;
end



J = (1/(-m)) * sum(sum((y1 .* log(p)) + ((1-y1) .* (log(1-p)))));

J = J + ((lambda/(2*m)) * (sum(sum((Theta1(1:hidden_layer_size, 2:(input_layer_size+1)) .* Theta1(1:hidden_layer_size, 2:(input_layer_size+1))))) +  sum(sum((Theta2(1:num_labels, 2:(hidden_layer_size+1)) .* Theta2(1:num_labels, 2:(hidden_layer_size+1))))) ));

D1 = zeros(size(Theta1)); % 25x401
D2 = zeros(size(Theta2)); % 10x26
X = tempX;
for i = 1:size(y, 1)
  y1 = zeros(num_labels, 1);
  y1(y(i, 1), 1) = 1;
  a1 = (X(i,:))'; % 400x1
  z2 = Theta1 * [1; a1]; % 25x1
  a2 = sigmoid(z2); % 25x1
  z3 = Theta2 * [1; a2]; % 10x1
  a3 = sigmoid(z3); % 10x1
  delta3 = a3 - y1; % 10x1
  delta2 = (Theta2' * delta3) .* sigmoidGradient([1; z2]); %26x1
  D1 = D1 + (delta2(2:(size(delta2, 1))) * [1; a1]'); %25x1 x 1x401 = 25x401
  D2 = D2 + (delta3 * [1; a2]'); %10x1 x 1x26 = 10x26
end

Theta1_grad = D1 / m; % 25x401
Theta2_grad = D2 / m; % 10x26
t1 = (lambda / m) * Theta1;
t1(:, 1) = 0;
Theta1_grad = Theta1_grad + t1;
t2 = (lambda / m) * Theta2;
t2(:, 1) = 0;
Theta2_grad = Theta2_grad + t2;
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
