x1 = 1:0.5:20;
x2 = x1+3.*rand(size(x1));
plot(x1,x2,'x');
z = zeros(size(x1));
z(1:7)=0;
z(8:21)=1;
z(22:end)=2;


X = [[x1];[x2]]'
m=length(z);
X = [ones(m, 1), X];
theta = zeros(size(1,:),m);
for i = 1:400
  J = (1/m)

