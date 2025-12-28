clc; clearvars; close all;

x = 0:6; 
y = [0, 0.8415, 0.9093, 0.1411, -0.7568, -0.9589, -0.2794];

N = length(x);
p = 0; %stores the final polynomial

for i=0:N-1
    temp = y(i+1); %each temporary term that will be added to the polynomial at the end
    for j=0:N-1
        if i~=j
            term = [1 -x(j+1)]/(x(i+1) - x(j+1));
            temp = conv(temp, term); 
        end
    end
    p = p + temp;
end

%making densely data points for the final plot

xq = linspace(min(x), max(x), 1000);
yq = polyval(p, xq);
plot(x, y, "o", LineWidth=2); hold on; plot(xq, yq, LineWidth=2);
title("Lagrange's Interpolation");
legend("Datapoints", "Interpolated curve");