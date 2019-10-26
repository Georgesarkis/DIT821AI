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
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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

%front propogation
a1 = [ones(m,1) X]; % add the baios +1
a2 = sigmoid(a1*Theta1');   %calculate hiddent layer
a2 = [ones(m,1) a2]; % add the baios in hidden layer +1
a3 = sigmoid(a2*Theta2'); % calculate output layer
y_mat = zeros(m,num_labels); %genereate y_mat which is y(k)
for i = 1:m,
    y_mat(i,y(i)) = 1;
end;

J = 1 / m * sum(sum( -y_mat .* log(a3) - (1-y_mat) .* log(1- a3))); 

reg = lambda / 2 / m * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2))); % regeerize cost funcaition to avoid overfiting
J = J + reg;


%back propogation
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
for t = 1:m,
	a1t = a1(t,:); %Set the input layer?s values (a(1)) to the t-th training example x(t).
	a2t = a2(t,:); %Set the input layer?s values (a(1)) to the t-th training example x(t).
	a3t = a3(t,:); %Set the input layer?s values (a(1)) to the t-th training example x(t).
	yt = y_mat(t,:); %Set the output layer?s values (y_mat(1)) to the t-th training example x(t).
	d3 = a3t - yt; %output unit ?k(3) = (ak(3) yk),
	d2 = Theta2'*d3' .* sigmoidGradient([1;Theta1 * a1t']); %1;Theta1 for bios +1, others for instruction nur 3
	delta1 = delta1 + d2(2:end)*a1t; %(2:end) to skip the bios +1: calculate delta between input and hidden layer
	delta2 = delta2 + d3' * a2t; % calculate theta between hidden layer and output layer
end;
 % https://medium.com/@shrutijadon10104776/why-we-dont-use-bias-in-regularization-5a86905dfcd6
 % skipping bios in regularization 
Theta1_grad = 1/m * delta1 + (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)]; % regularize neural networks for delta1
Theta2_grad = 1/m * delta2 + (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)]; % regularize neural networks for delta2

% returning final gradel matrix
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
