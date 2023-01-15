function [Theta,Y] = Newton(f,x0,X)
k=0;
Theta=zeros(2001,2); Y=[];
grad=gradient(f,X);
hess=hessian(f,X);
x=x0;
alpha=0.001;
while k<=2000
    Theta(k+1,:)=x;
    Y=[Y double(subs(f,X,x))];
    x=x-alpha*(double(subs(hess,X,x))\double(subs(grad,X,x)))';
    k=k+1;
end
end