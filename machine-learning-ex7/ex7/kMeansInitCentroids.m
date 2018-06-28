function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

%randomly reorder the indices of examples (permutation)
randidx = randperm(size(X,1));

%Take the first K examples from shuffled indices as centroids
%note: X is not shuffled; only a copy of indices are shuffled 
centroids = X(randidx(1:K),:);


% =============================================================

end

