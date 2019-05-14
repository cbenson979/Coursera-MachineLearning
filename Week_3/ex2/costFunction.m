function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

%%% Evaluate the J and gradient
J = (1./m).*sum(-1.*y'.*log(sigmoid(theta'*X'))-(1-y').*(log(1-sigmoid(theta'*X'))));
grad = (1./m).*((sigmoid(theta'*X')-y')*X)';

% =============================================================

end
