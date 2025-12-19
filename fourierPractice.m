clc; clearvars; close all;

fs = 2000;
T = 1;
t = 0:1/fs:T-(1/fs);

signal = 10*sin(2*pi*60*t) + 5*sin(2*pi*160*t);
hiss = 2*cos(2*pi*600*t + pi/6);
randNoise = 1.5*randn(size(t));

sig = signal + hiss + randNoise;

% now we have to get the frequency domain of sig

f = fft(sig);

% normalizing the frequency domain
N = length(sig);
p = abs(f/N);
p1 = p(1:floor(N/2+1));
p1(2:end-1) = 2*p1(2:end-1);
f1 = fs*(0:(N/2))/N;
plot(f1, p1, LineWidth=2);
title("Normalized fft amplitude sprectrum");

% applying the filter

fullFrequencyMap = (0:N-1)*(fs/N);
mask = (fullFrequencyMap <= 200) | (fullFrequencyMap >= (fs-200));
filteredSignal = mask.*f;
recoveredSignal = real(ifft(filteredSignal));

