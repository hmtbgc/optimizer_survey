%马鞍面函数创建
X=sym('x',[1 2]);
f=X(1)^2-X(2)^2;
%马鞍面绘制
x=-2:0.01:2;
y=-2:0.01:2;
[x1,y1]=meshgrid(x,y);
z=x1.^2-y1.^2;
mesh(x1,y1,z);
str='(0,0,0)';
text(0,0,0.5,str);
title('{$z=x^{2}-y^{2}$}','Interpreter','latex');
xlabel('{$x$}','interpreter','latex');
ylabel('{$y$}','Interpreter','latex');
zlabel('{$z$}',"Interpreter","latex");


%不同算法结果对比
[X_SGD,Z_SGD]=SGD(f,[1,1],X);
[X_SGD_M,Z_SGD_M]=SGD_M(f,[1,1],X);
[X_SGD_NM,Z_SGD_NM]=SGD_NM(f,[1,1],X);
[X_Adagrad,Z_Adagrad]=Adagrad(f,[1,1],X);
[X_RMSprop,Z_RMSprop]=RMSprop(f,[1,1],X);
[X_Adam,Z_Adam]=Adam(f,[1,1],X);
[X_Adamax,Z_Adamax]=Adamax(f,[1,1],X);
[X_NAdam,Z_NAdam]=NAdam(f,[1,1],X);
[X_Newton,Z_Newton]=Newton(f,[1,1],X);
[X_BFGS,Z_BFGS]=BFGS(f,[1,1],X);
[X_BFGS_M,Z_BFGS_M]=BFGS_M(f,[1,1],X);
[X_Hess_Free,Z_Hess_Free]=Hess_Free(f,[1,1],X);
[X_NAdam_Newton,Z_NAdam_Newton]=NAdam_Newton(f,[1,1],X);
Dist_SGD=Dist(X_SGD,Z_SGD);
Dist_SGD_M=Dist(X_SGD_M,Z_SGD_M);
Dist_SGD_NM=Dist(X_SGD_NM,Z_SGD_NM);
Dist_Adagrad=Dist(X_Adagrad,Z_Adagrad);
Dist_RMSprop=Dist(X_RMSprop,Z_RMSprop);
Dist_Adam=Dist(X_Adam,Z_Adam);
Dist_Adamax=Dist(X_Adamax,Z_Adamax);
Dist_NAdam=Dist(X_NAdam,Z_NAdam);
Dist_Newton=Dist(X_Newton,Z_Newton);
Dist_BFGS=Dist(X_BFGS,Z_BFGS);
Dist_BFGS_M=Dist(X_BFGS_M,Z_BFGS_M);
Dist_Hess_Free=Dist(X_Hess_Free,Z_Hess_Free);
Dist_NAdam_Newton=Dist(X_NAdam_Newton,Z_NAdam_Newton);
%一阶方法
plot(1:2001,Z_SGD,'r','LineWidth',1.5);
axis([1 2001 -10^3 0])
xlabel('time(iterations)');
ylabel("error");
title('Error comparison among 1-ordered optimizers (to saddle point)');
hold on;
plot(1:2001,Z_SGD_M,'g','LineWidth',1.5);
hold on;
plot(1:2001,Z_SGD_NM,'y','LineWidth',1.5);
hold on;
plot(1:2001,Z_Adagrad,'c','LineWidth',1.5);
hold on;
plot(1:2001,Z_RMSprop,'b','LineWidth',1.5);
hold on;
plot(1:2001,Z_Adam,'m','LineWidth',1.5);
hold on;
plot(1:2001,Z_Adamax,'color',[1,0.5,0],'LineWidth',1.5);
hold on;
plot(1:2001,Z_NAdam,'color',[0.5,0,0],'LineWidth',1.5);
legend('SGD','SGD+Momentum','SGD+Nesterov Momentum','Adagrad','RMSprop','Adam','Adamax','NAdam','FontSize',8);

plot(1:2001,Dist_SGD,'r','LineWidth',1.5);
axis([1 2001 0 10^4])
xlabel('time(iterations)');
ylabel("distance");
title('Distance comparison among 1-ordered optimizers (to saddle point)');
hold on;
plot(1:2001,Dist_SGD_M,'g','LineWidth',1.5);
hold on;
plot(1:2001,Dist_SGD_NM,'y','LineWidth',1.5);
hold on;
plot(1:2001,Dist_Adagrad,'c','LineWidth',1.5);
hold on;
plot(1:2001,Dist_RMSprop,'b','LineWidth',1.5);
hold on;
plot(1:2001,Dist_Adam,'m','LineWidth',1.5);
hold on;
plot(1:2001,Dist_Adamax,'color',[1,0.5,0],'LineWidth',1.5);
hold on;
plot(1:2001,Dist_NAdam,'color',[0.5,0,0],'LineWidth',1.5);
legend('SGD','SGD+Momentum','SGD+Nesterov Momentum','Adagrad','RMSprop','Adam','Adamax','NAdam','FontSize',8)

plot(1:2001,Z_Adagrad,'c','LineWidth',1.5);
axis([1 2001 -10 0])
xlabel('time(iterations)');
ylabel("error");
title('Error comparison among 1-ordered optimizers (to saddle point)');
hold on;
hold on;
plot(1:2001,Z_RMSprop,'b','LineWidth',1.5);
hold on;
plot(1:2001,Z_Adam,'m','LineWidth',1.5);
hold on;
plot(1:2001,Z_Adamax,'color',[1,0.5,0],'LineWidth',1.5);
hold on;
plot(1:2001,Z_NAdam,'color',[0.5,0,0],'LineWidth',1.5);
legend('Adagrad','RMSprop','Adam','Adamax','NAdam','FontSize',8);

plot(1:2001,Dist_Adagrad,'c','LineWidth',1.5);
axis([1 2001 0 20])
xlabel('time(iterations)');
ylabel("distance");
title('Distance comparison among 1-ordered optimizers (to saddle point)');
hold on;
plot(1:2001,Dist_RMSprop,'b','LineWidth',1.5);
hold on;
plot(1:2001,Dist_Adam,'m','LineWidth',1.5);
hold on;
plot(1:2001,Dist_Adamax,'color',[1,0.5,0],'LineWidth',1.5);
hold on;
plot(1:2001,Dist_NAdam,'color',[0.5,0,0],'LineWidth',1.5);
legend('Adagrad','RMSprop','Adam','Adamax','NAdam','FontSize',8)

%二阶方法
plot(1:2001,Z_Newton,'r','LineWidth',1.5);
axis([1 2001 -10^4 0])
xlabel('time(iterations)');
ylabel("error");
title('Error comparison among 2-ordered optimizers (to saddle point)');
hold on;
plot(1:2001,Z_BFGS,'g','LineWidth',1.5);
hold on;
plot(1:2001,Z_BFGS_M,'y','LineWidth',1.5);
hold on;
legend('Newton','BFGS','BFGS+Momentum','FontSize',8);

plot(1:2001,Dist_Newton,'r','LineWidth',1.5);
axis([1 2001 0 10^4]);
xlabel('time(iterations)');
ylabel("distance");
title('Distance comparison among 2-ordered optimizers (to saddle point)');
hold on;
plot(1:2001,Dist_BFGS,'g','LineWidth',1.5);
hold on;
plot(1:2001,Dist_BFGS_M,'y','LineWidth',1.5);
hold on;
legend('Newton','BFGS','BFGS+Momentum','FontSize',8);

%融合一阶二阶方法后与原方法比较
plot(1:2001,Z_NAdam,'c','LineWidth',1.5);
axis([1 2001 -150 0]);
xlabel('time(iterations)');
ylabel("error");
title('Error of improved method (to saddle point)');
hold on;
plot(1:2001,Z_Newton,'b','LineWidth',1.5);
hold on;
plot(1:2001,Z_NAdam_Newton,'m','LineWidth',1.5);
legend('NAdam','Newton','NAdam+Newton','FontSize',8);

plot(1:2001,Dist_NAdam,'c','LineWidth',1.5);
axis([1 2001 0 150]);
xlabel('time(iterations)');
ylabel("distance");
title('Distance of improved method (to saddle point)');
hold on;
plot(1:2001,Dist_Newton,'b','LineWidth',1.5);
hold on;
plot(1:2001,Dist_NAdam_Newton,'m','LineWidth',1.5);
legend('NAdam','Newton','NAdam+Newton','FontSize',8);










