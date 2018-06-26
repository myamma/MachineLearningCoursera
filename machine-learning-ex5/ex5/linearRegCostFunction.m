function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0.0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

diff = X*theta - y; % diff is mx1
J = (1/2/m)*(sum(diff.^2));
J = J + (lambda/2/m)*sum(theta(2:end).^2);

for i = 1:size(theta)
  if i == 1  
    grad(i) = (1/m)*sum(diff.*X(:,i));
  else
    grad(i) = (1/m)*sum(diff.*X(:,i)) + theta(i)*lambda/m;
  end
end

%grad = X'*(H - y) / m + lambda*[0;theta(2:end)] / m;

% =========================================================================


end
