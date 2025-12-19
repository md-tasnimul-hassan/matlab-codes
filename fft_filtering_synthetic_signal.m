clc; clearvars; close all;

fs = 5000; 
T = 2;
t = 0:1/fs:T-(1/fs);

% making the signals 

sig = 5*sin(2*pi*120*t + pi/6);
noise = 3*cos(2*pi*400*t) + sin(2*pi*1000*t-pi/3);
gausianNoise = 5*randn(size(t));
signal = sig+noise+gausianNoise;

% doing fft and plotting the frequency spectrum 

f = fft(signal);
N = length(f);
% normalizing 
m = abs(f/N);
% taking the real part of fourier
m = m(1:floor(N/2+1));
% scalling
m(2:end-1) = 2*m(2:end-1);
% making frequency scale 
f1 = fs*(0:(N/2))/N;

% visualizing the frequency domain
figure;
plot(f1, m)
title("Frequency domain");


% filtering and constructing the real signal 
frequencyMap = (0:N-1)*fs/N;
mask = (frequencyMap >= 110 & frequencyMap <= 130) | (frequencyMap <= (fs-110) & frequencyMap >= (fs-130));
f = f.*mask;
reconstructedSignal = real(ifft(f));

% final demonstration
figure;
subplot(2,1,1);
plot(t, signal, LineWidth=2);
xlim([0 0.1]);
title("Input signal");
subplot(2,1,2);
plot(t, sig,"r-", t, reconstructedSignal,"k--", LineWidth=2);
xlim([0 0.1]);
title("Output signal");
legend("Real signal", "Reconstructed signal");
