function [Theta,Y] = BFGS_M(f,x0,X)
k=0;
x=x0;
gamma=0.9;
alpha=0.001;
m=zeros(1,2);
H=eye(2);
Theta=zeros(2001,2); Y=[];
grad=gradient(f,X);
while k<=2000
    Theta(k+1,:)=x;
    Y=[Y double(subs(f,X,x))];
    x1=x;
    m=gamma*m+alpha*double(subs(grad,X,x))';
    x=x-m;
    x2=x;
    s=x2-x1;
    y=double(subs(grad,X,x2))'-double(subs(grad,X,x1))';
    H=H-(H*s'*s*H)/(s*H*s')+(y'*y)/(y*s');
    k=k+1;
end
end