function [Theta,Y] = Hess_Free(f,x0,X)
k=0;
x=x0;
gamma=0.001;
Theta=zeros(2001,2); Y=[];
grad=gradient(f,X);
    function [x1]=H_F(x,d)
        x1=(double(subs(grad,X,x+10^(-10)*d'))-double(subs(grad,X,x)))*10^10;
    end
    function [x2]=CG(x,g)
        ng=norm(g);
        CG_tol=min(0.5,sqrt(ng))*ng;
        r=g;
        p=-r;
        x2=0;
        for iter=1:2000
            rr=r'*r;
            Bp=H_F(x,p);
            alpha=rr/(p'*Bp);
            x2=x2+alpha*p;
            r=r+alpha*Bp;
            nrl=norm(r);
            if nrl<=CG_tol
                break;
            end
            beta=nrl^2/rr;
            p=-r+beta*p;
        end
    end
while k<=2000
    Theta(k+1,:)=x;
    Y=[Y double(subs(f,X,x))];
    g=double(subs(grad,X,x));
    x=x-gamma*CG(x,g)';
    k=k+1;
end
end