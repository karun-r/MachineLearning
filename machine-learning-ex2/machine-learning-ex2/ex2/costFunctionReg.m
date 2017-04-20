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

thetaT = theta';
hyp = sigmoid(thetaT*X');
for in = 1:m
    J = J + ((-y(in)*log(hyp(in)))- ((1-y(in))*log(1-hyp(in))));
end
% =============================================================
J = J/m;
lam = 0;
[l,n]=size(X(1,:));
for i = 2:n
    lam = lam + theta(i)^2;
end
J = (lambda/(2*m))*lam + J;

% =============================================================
for nh = 1:size(theta)
    diff = 0;
    for inh =  1:m
        diff = diff + (1/m)*((hyp(inh)-y(inh))*X(inh,nh));
    end
    if(nh>1)
        grad(nh) = diff + (lambda/m)*theta(nh);
    else
        grad(nh) = diff ;
    end
end
end
