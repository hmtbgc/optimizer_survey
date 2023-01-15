function [Theta,Y] = BFGS(f,x0,X)
k=0;
x=x0;
alpha=0.001;
H=eye(2);
Theta=zeros(2001,2); Y=[];
grad=gradient(f,X);
while k<=2000
    Theta(k+1,:)=x;
    Y=[Y double(subs(f,X,x))];
    x1=x;
    x=x-alpha*(H*double(subs(grad,X,x)))';
    x2=x;
    s=x2-x1;
    y=double(subs(grad,X,x2))'-double(subs(grad,X,x1))';
    rho=1/(s*y');
    H=(eye(2)-rho*s'*y)*H*(eye(2)-rho*y'*s)+rho*(s'*s);
    k=k+1;
end
end