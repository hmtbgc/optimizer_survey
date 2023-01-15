function [Theta,Y] = Adagrad(f,x0,X)
k=0;
x=x0;
Theta=zeros(2001,2); Y=[];
m=0;
alpha=0.001;
epsilon=10^(-8);
grad=gradient(f,X);
while k<=2000
    Theta(k+1,:)=x;
    Y=[Y double(subs(f,X,x))];
    m=m+norm(double(subs(grad,X,x)))^2;
    x=x-alpha/sqrt(m+epsilon)*double(subs(grad,X,x))';
    k=k+1;
end
end