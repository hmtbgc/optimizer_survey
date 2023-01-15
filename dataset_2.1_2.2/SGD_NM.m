function [Theta,Y] = SGD_NM(f,x0,X)
k=0;
x=x0;
Theta=zeros(2001,2); Y=[];
m=zeros(1,2);
alpha=0.001;
gamma=0.9;
grad=gradient(f,X);
while k<=2000
    Theta(k+1,:)=x;
    Y=[Y double(subs(f,X,x))];
    m=gamma*m+alpha*double(subs(grad,X,x-gamma*m))';
    x=x-m;
    k=k+1;
end
end