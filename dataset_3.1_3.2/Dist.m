function [dist] = Dist(mat,Y)
p=[0,0,0];
for i=1:length(Y)
    dist(i)=norm([mat(i,:),Y(i)]-p);
end
end