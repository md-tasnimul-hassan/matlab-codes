clc; clearvars; close all; 
x = [-12, -10, -8, -6, -4, -2, 0, 1, 2, 3, 4, 5, 6];
y = [-0.005, -0.028, -0.16, -0.808, -2.757, -4.76, 0, -0.5, -1, 0.01, 3, 5, 10];
plot(x,y,"o",LineWidth=2);
hold on;
for degree = 2:9
    N = length(x);
    n = degree + 1;
    A = zeros(n,n);
    B = zeros(n,1);
    
    for r = 1:n
        for c = 1:n
            power = (r-1) + (c-1);
            A(r,c) = sum(x.^power);
        end
    end
    
    for r=1:n
        power = r-1;
        B(r) = sum((x.^power).*y);
    end
    
    coef = A\B;
    xq = linspace(min(x), max(x), 1000);
    yq = zeros(size(xq));
    for i=1:n
        yq = yq + coef(n-i+1)*(xq.^(n-i));
    end
    plot(xq, yq, LineWidth=2);
    % sprintf("Degree %d", degree)
end
legend show
