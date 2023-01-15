%目标函数的构造
X=sym("x",[1 2]);
f=log(1+(1.5-X(1)+X(1)*X(2))^2+(2.25-X(1)+X(1)*X(2)^2)^2+(2.625-X(1)+X(1)*X(2)^3)^2)/10;
%二元函数图像绘制
x=0:0.01:5;
y=-3:0.01:2;
[x1,y1]=meshgrid(x,y);
z=log(1+(1.5-x1+x1.*y1).^2+(2.25-x1+x1.*y1.^2).^2+(2.625-x1+x1.*y1.^3).^2)/10;
mesh(x1,y1,z);
str='(3,0.5,0)';
text(3,0.5,0,str);
title('{$f(x,y)=\frac{log(1+(1.5-x+xy)^{2}+(2.25-x+xy^{2})^{2}+(2.625-x+xy^{3})^{2}}{10}$}','Interpreter','latex');
xlabel('{$x$}','interpreter','latex');
ylabel('{$y$}','Interpreter','latex');
zlabel('{$f(x,y)$}',"Interpreter","latex");


%不同算法结果对比
[Theta_SGD,Y_SGD]=SGD(f,[2,0],X);
[Theta_SGD_M,Y_SGD_M]=SGD_M(f,[2,0],X);
[Theta_SGD_NM,Y_SGD_NM]=SGD_NM(f,[2,0],X);
[Theta_Adagrad,Y_Adagrad]=Adagrad(f,[2,0],X);
[Theta_RMSprop,Y_RMSprop]=RMSprop(f,[2,0],X);
[Theta_Adam,Y_Adam]=Adam(f,[2,0],X);
[Theta_Adamax,Y_Adamax]=Adamax(f,[2,0],X);
[Theta_NAdam,Y_NAdam]=NAdam(f,[2,0],X);
[Theta_Newton,Y_Newton]=Newton(f,[2,0],X);
[Theta_BFGS,Y_BFGS]=BFGS(f,[2,0],X);
[Theta_BFGS_M,Y_BFGS_M]=BFGS_M(f,[2,0],X);
[Theta_Hess_Free,Y_Hess_Free]=Hess_Free(f,[2,0],X);
[Theta_NAdam_Newton,Y_NAdam_Newton]=NAdam_Newton(f,[2,0],X);
dist_SGD=Dist(Theta_SGD,Y_SGD);
dist_SGD_M=Dist(Theta_SGD_M,Y_SGD_M);
dist_SGD_NM=Dist(Theta_SGD_NM,Y_SGD_NM);
dist_Adagrad=Dist(Theta_Adagrad,Y_Adagrad);
dist_RMSprop=Dist(Theta_RMSprop,Y_RMSprop);
dist_Adam=Dist(Theta_Adam,Y_Adam);
dist_Adamax=Dist(Theta_Adamax,Y_Adamax);
dist_NAdam=Dist(Theta_NAdam,Y_NAdam);
dist_Newton=Dist(Theta_Newton,Y_Newton);
dist_BFGS=Dist(Theta_BFGS,Y_BFGS);
dist_BFGS_M=Dist(Theta_BFGS_M,Y_BFGS_M);
dist_Hess_Free=Dist(Theta_Hess_Free,Y_Hess_Free);
dist_NAdam_Newton=Dist(Theta_NAdam_Newton,Y_NAdam_Newton);
%一阶方法
plot(1:2001,Y_SGD,'r','LineWidth',1.5);
axis([1 2001 0 0.055]);
xlabel('time(iterations)');
ylabel("error");
title('Error comparison among 1-ordered optimizers');
hold on;
plot(1:2001,Y_SGD_M,'g','LineWidth',1.5);
hold on;
plot(1:2001,Y_SGD_NM,'y','LineWidth',1.5);
hold on;
plot(1:2001,Y_Adagrad,'c','LineWidth',1.5);
hold on;
plot(1:2001,Y_RMSprop,'b','LineWidth',1.5);
hold on;
plot(1:2001,Y_Adam,'m','LineWidth',1.5);
hold on;
plot(1:2001,Y_Adamax,'color',[1,0.5,0],'LineWidth',1.5);
hold on;
plot(1:2001,Y_NAdam,'color',[0.5,0,0],'LineWidth',1.5);
legend('SGD','SGD+Momentum','SGD+Nesterov Momentum','Adagrad','RMSprop','Adam','Adamax','NAdam','FontSize',8);

plot(1:2001,dist_SGD,'r','LineWidth',1.5);
axis([1 2001 0 1.2]);
xlabel('time(iterations)');
ylabel("distance");
title('Distance comparison among 1-ordered optimizers');
hold on;
plot(1:2001,dist_SGD_M,'g','LineWidth',1.5);
hold on;
plot(1:2001,dist_SGD_NM,'y','LineWidth',1.5);
hold on;
plot(1:2001,dist_Adagrad,'c','LineWidth',1.5);
hold on;
plot(1:2001,dist_RMSprop,'b','LineWidth',1.5);
hold on;
plot(1:2001,dist_Adam,'m','LineWidth',1.5);
hold on;
plot(1:2001,dist_Adamax,'color',[1,0.5,0],'LineWidth',1.5);
hold on;
plot(1:2001,dist_NAdam,'color',[0.5,0,0],'LineWidth',1.5);
legend('SGD','SGD+Momentum','SGD+Nesterov Momentum','Adagrad','RMSprop','Adam','Adamax','NAdam','FontSize',8)


%二阶方法
plot(1:2001,Y_Newton,'r','LineWidth',1.5);
axis([1 2001 0 0.055]);
xlabel('time(iterations)');
ylabel("error");
title('Error comparison among 2-ordered optimizers');
hold on;
plot(1:2001,Y_BFGS,'g','LineWidth',1.5);
hold on;
plot(1:2001,Y_BFGS_M,'y','LineWidth',1.5);
legend('Newton','BFGS','BFGS+Momentum','FontSize',8);

plot(1:2001,Y_Hess_Free,'c','LineWidth',1.5);
axis([0 2001 0 0.12]);
xlabel('time(iterations)');
ylabel("error");
title('Error of Hessian-Free');

plot(1:2001,dist_Newton,'r','LineWidth',1.5);
axis([1 2001 0 1.2])
xlabel('time(iterations)');
ylabel("distance");
title('Distance comparison among 2-ordered optimizers');
hold on;
plot(1:2001,dist_BFGS,'g','LineWidth',1.5);
hold on;
plot(1:2001,dist_BFGS_M,'y','LineWidth',1.5);
legend('Newton','BFGS','BFGS+Momentum','FontSize',8);

plot(1:2001,dist_Hess_Free,'c','LineWidth',1.5);
axis([0 2001 0 1.8]);
xlabel('time(iterations)');
ylabel("distance");
title('Distance of Hessian-Free');

%融合一阶二阶方法后与原方法比较
plot(1:2001,Y_NAdam,'c','LineWidth',1.5);
axis([1 2001 0 0.055]);
xlabel('time(iterations)');
ylabel("error");
title('Error of improved method');
hold on;
plot(1:2001,Y_Newton,'b','LineWidth',1.5);
hold on;
plot(1:2001,Y_NAdam_Newton,'m','LineWidth',1.5);
legend('NAdam','Newton','NAdam+Newton','FontSize',8);

plot(1:2001,dist_NAdam,'c','LineWidth',1.5);
axis([1 2001 0 1.2]);
xlabel('time(iterations)');
ylabel("distance");
title('Distance of improved method');
hold on;
plot(1:2001,dist_Newton,'b','LineWidth',1.5);
hold on;
plot(1:2001,dist_NAdam_Newton,'m','LineWidth',1.5);
legend('NAdam','Newton','NAdam+Newton','FontSize',8);