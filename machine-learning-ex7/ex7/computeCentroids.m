function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n+1);
% disp('Size of X'); disp(size(X));      300x2
% disp('Size of idx'); disp(size(idx));  300x1
for i = 1:m
  centroids(idx(i), 1:n) = centroids(idx(i), 1:n) + X(i, :);
  centroids(idx(i), n+1) = centroids(idx(i), n+1) + 1;
end
for i = 1:K
  centroids(i, :) = centroids(i, :) / centroids(i, n+1);
end

centroids = centroids(:, 1:n);
% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%








% =============================================================


end

