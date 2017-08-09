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

%part-1
%将类转换为向量
Y=[];
E =eye(num_labels);
for i=1:num_labels
    Y0=find(y==i);
    Y(Y0,:)=repmat(E(i,:),size(Y0,1),1);
end

%forward  
% 
X =[ones(m,1) X];
a2 = sigmoid(X*Theta1');
a2 = [ones(m,1) a2];
a3 =sigmoid(a2*Theta2');

temp1 =  Theta1(:,2:end).^2;   % 先把theta(1)拿掉，不参与正则化
temp2 =  Theta2(:,2:end).^2;
cost = -Y.*log(a3)-(1-Y).*(log(1-a3));
J = (1/m).*sum(cost(:))+(lambda/(2*m))*(sum(temp1(:))+sum(temp2(:)));

%Gradient
delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));

for t=1:m
  %遍历每一个sample
  a1 = X(t,:)';
  z2 = Theta1*a1;
  a2 = sigmoid(z2);
  a2 = [1;a2];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  
  %计算输出层的误差
  err3 = zeros(num_labels,1);
  for k=1:num_labels
    err3(k) = a3(k)-(y(t)==k);
  end
  
 %第二个隐含层
err2 = Theta2'*err3;
%除去第一行，因为第一行的元素不参与往后的一层运算。只参与前向的
%所以省去
err2 = err2(2:end) .*sigmoidGradient(z2);
delta_2 =delta_2+err3*a2';
delta_1 = delta_1+err2*a1';
end
  %把第一行设置为0,因为在计算的时候第一行不加上theta
Theta1_temp=[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_temp=[zeros(size(Theta2,1),1) Theta2(:,2:end)];
  
Theta1_grad =1/m* delta_1 + lambda/m*Theta1_temp;
Theta2_grad =1/m* delta_2 + lambda/m*Theta2_temp;







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
