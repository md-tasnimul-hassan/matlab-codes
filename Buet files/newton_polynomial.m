clc; clearvars; close all;
x = 0:6; 
y = [0, 0.8415, 0.9093, 0.1411, -0.7568, -0.9589, -0.2794];
N = length(x);
D = zeros(N);
D(:,1) = y';
for i=2:N
    for j=i:N
        D(j,i) = (D(j,i-1)-D(j-1,i-1))/(x(j)-x(j-i+1));
    end
end
d = diag(D);
xq = linspace(min(x),max(x),1000);
yq = d(N)*ones(size(xq));
for i=(N-1):-1:1
    yq = d(i)+ (xq-x(i)).*yq;
end
plot(x,y,"o",LineWidth=2);
hold on;
plot(xq, yq, LineWidth=2);
title("Newton Interpolation");
legend("Data points", "Interpolated curve");