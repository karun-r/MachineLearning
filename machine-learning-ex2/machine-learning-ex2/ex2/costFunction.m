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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
thetaT = theta';
hyp = sigmoid(thetaT*X');
hyc = size(hyp);
for in = 1:m
    J = J + ((-y(in)*log(hyp(in)))- ((1-y(in))*log(1-hyp(in))));
end
% =============================================================
J = J/m;
for nh = 1:size(theta)
    diff = 0;
    for inh =  1:m
        diff = diff + (1/m)*((hyp(inh)-y(inh))*X(inh,nh));
    end
    grad(nh) = diff;
end
