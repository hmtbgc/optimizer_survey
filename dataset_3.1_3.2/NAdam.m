function [Theta,Y] = NAdam(f,x0,X)
k=0;
x=x0;
Theta=zeros(2001,2); Y=[];
m=zeros(1,2);
v=0;
alpha=0.001;
beta1=0.9;
beta2=0.999;
epsilon=10^(-8);
grad=gradient(f,X);
while k<=2000
    Theta(k+1,:)=x;
    Y=[Y double(subs(f,X,x))];
    m=beta1*m+(1-beta1)*double(subs(grad,X,x))';
    m1=m/(1-beta1);
    m2=beta1*m1+(1-beta1)*double(subs(grad,X,x))';
    v=beta2*v+(1-beta2)*norm(double(subs(grad,X,x)))^2;
    x=x-alpha*sqrt(1-beta2)/(1-beta1)*m2/(sqrt(v)+epsilon);
    k=k+1;
end
end