function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    %ensures simultaneous update
    diff = X*theta - y;

##    %loop implmentation (method 1), update one feature at a time
##    for i = 1:n
##      diffSum = (diff)'*X(:,i); %diffSum is a constant 1x1
##      theta(i,1) = theta(i,1) - alpha*(1/m)*diffSum;    
##    end

    %vector implementation (method 2), update all features at the same time
    diffSum = ((diff)'*X)'; %vectorized diffSum after transpose is (n+1)x1 ~ nx1 matrix
    theta = theta - (alpha/m)*diffSum;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
