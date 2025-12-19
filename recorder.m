clc; clearvars; close all;
% Setup Recording Parameters
Fs = 16000;      % 16 kHz Sampling Rate
nBits = 16;      % 16-bit
nChannels = 1;   % Mono
duration = 2.5;    % Duration in seconds

% Create the recorder object
recObj = audiorecorder(Fs, nBits, nChannels);

%User Interaction
disp('------------------------------------------');
disp('   Voice Password Data Collection');
disp('------------------------------------------');

% The 's' argument treats the input as a string, preventing errors if you just hit Enter
input('>> Press [ENTER] to start recording...', 's'); 

% Recording
disp('Recording NOW... Speak "My voice is my password"');
recordblocking(recObj, duration); % Records for exactly 4 seconds
disp('Recording Finished.');

% Playback and Save
% 1. Get data
audioData = getaudiodata(recObj);

% 2. Play back to verify
disp('Playing back...');
play(recObj);

% 3. Save to file
filename = 'test.wav';
audiowrite(filename, audioData, Fs);
fprintf('File saved as: %s\n', filename);

% 4. Quick Plot (Optional Check)
plot(audioData);
title('Recorded Audio');
axis tight;