clc; clearvars; close all;

x = 0:6;
y = [0, 0.8415, 0.9093, 0.1411, -0.7568, -0.9589, -0.2794];

N = length(x);
D = zeroes(N);
D(:,1) = y';

for i=2:N
    for j=i:N
        % add codes
    end
end