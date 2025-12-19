clc; clearvars; close all;

[signal, fs] = audioread("sample.wav");
if isrow(signal)
    signal = signal';
end

% --- Step A: Pre-emphasis ---
% High-pass filter to balance the spectrum (H(z) = 1 - 0.97z^-1)
alpha = 0.97;
signal = filter([1 -alpha], 1, signal);

% --- Step B: Safe Frequency Domain Filtering ---
% Using a smooth "Tapered" mask to avoid ringing artifacts
N = length(signal);
f_fft = fft(signal);
freqs = (0:N-1)*fs/N;

% Define passband: 300Hz - 3400Hz
mask = (freqs >= 300 & freqs <= 3400) | (freqs >= fs-3400 & freqs <= fs-300);
mask = double(mask'); % Convert logical to double

% Apply a simple moving average to the mask to "soften" the edges
% This prevents the Gibbs Phenomenon (ringing)
smooth_mask = conv(mask, ones(50,1)/50, 'same'); 

cleanSignal = real(ifft(f_fft .* smooth_mask));

% --- Step C: Normalization ---
cleanSignal = cleanSignal / max(abs(cleanSignal));

% --- MFCC Section 1: Framing and Windowing ---

% Parameters
frameDuration = 0.025; % 25 ms
hopDuration = 0.010;   % 10 ms (step size)

% Convert durations from seconds to samples
L = round(frameDuration * fs); % Frame length in samples
S = round(hopDuration * fs);   % Step length in samples

% Calculate total number of frames
totalSamples = length(cleanSignal);
numFrames = floor((totalSamples - L) / S) + 1;

% Pre-allocate matrix for frames (each column is a frame)
% We do this to make the math faster
frames = zeros(L, numFrames);

% 1. Extract Frames
for i = 1:numFrames
    startIdx = (i-1)*S + 1;
    stopIdx = startIdx + L - 1;
    frames(:, i) = cleanSignal(startIdx:stopIdx);
end

% 2. Create Hamming Window manually
% Formula: w(n) = 0.54 - 0.46 * cos(2*pi*n / (L-1))
n = (0:L-1)';
hammingWin = 0.54 - 0.46 * cos(2 * pi * n / (L - 1));

% 3. Apply Window to all frames
% We multiply each column of the 'frames' matrix by the window
windowedFrames = frames .* repmat(hammingWin, 1, numFrames);

% --- Visualization ---
figure;
subplot(3,1,1); plot(cleanSignal); title('Cleaned Signal');
subplot(3,1,2); plot(frames(:, 10)); title('Frame #10 (Rectangular)');
subplot(3,1,3); plot(windowedFrames(:, 10)); title('Frame #10 (Hamming Windowed)');

% --- MFCC Section 2: Power Spectrum and Mel-Filterbank ---

% Parameters
NFFT = 512; % Standard FFT size for 25ms frames at 16-44.1kHz
numFilters = 26; % Standard number of Mel filters

% 1. Periodogram Power Spectrum
% Calculate FFT for each frame, take the magnitude, and square it
magnitudeSpectrum = abs(fft(windowedFrames, NFFT)); 
% Keep only the first half (positive frequencies)
magnitudeSpectrum = magnitudeSpectrum(1:NFFT/2+1, :);
powerSpectrum = (magnitudeSpectrum.^2) / NFFT;

% 2. Mel-Filterbank Design (Manual)
% Convert Hz to Mel: m = 2595 * log10(1 + f/700)
% Convert Mel to Hz: f = 700 * (10^(m/2595) - 1)

lowFreqHz = 300; 
highFreqHz = fs/2; % Nyquist frequency

% Convert Hz limits to Mel scale
lowMel = 2595 * log10(1 + lowFreqHz/700);
highMel = 2595 * log10(1 + highFreqHz/700);

% Create linearly spaced points in the Mel scale
melPoints = linspace(lowMel, highMel, numFilters + 2);

% Convert Mel points back to Hz
hzPoints = 700 * (10.^(melPoints / 2595) - 1);

% Convert Hz points to FFT bin indices
binPoints = floor((NFFT + 1) * hzPoints / fs);

% Construct the Filterbank Matrix
fbank = zeros(numFilters, NFFT/2 + 1);
for m = 2:numFilters + 1
    f_m_minus = binPoints(m-1);
    f_m = binPoints(m);
    f_m_plus = binPoints(m+1);
    
    for k = f_m_minus:f_m
        fbank(m-1, k+1) = (k - f_m_minus) / (f_m - f_m_minus);
    end
    for k = f_m:f_m_plus
        fbank(m-1, k+1) = (f_m_plus - k) / (f_m_plus - f_m);
    end
end

% 3. Apply Filterbank to Power Spectrum
% Result: Energy in each Mel band for every frame
filterbankEnergies = fbank * powerSpectrum;

% 4. Logarithm
% Human hearing is logarithmic; we also avoid log(0) with a tiny epsilon
filterbankEnergies = log(filterbankEnergies + 1e-10);

% --- Visualization ---
figure;
subplot(2,1,1); imagesc(fbank); title('Mel-Filterbank (Triangle Filters)');
ylabel('Filter Index'); xlabel('FFT Bin');
subplot(2,1,2); imagesc(filterbankEnergies); title('Log-Mel Energy');
ylabel('Filter Index'); xlabel('Frame Index');

% --- MFCC Section 3: MANUAL DCT and Final Feature Selection ---

[numBanks, numFrames] = size(filterbankEnergies);
numCEPS = 13; 

% 1. Create the DCT Matrix (Manual)
% N = numBanks (usually 26), M = numCEPS (usually 13)
N = numBanks;
M = numCEPS;
dctMatrix = zeros(M, N);

for k = 0:M-1
    % Normalization factor for orthonormality
    if k == 0
        alpha = sqrt(1/N);
    else
        alpha = sqrt(2/N);
    end
    
    for n = 0:N-1
        dctMatrix(k+1, n+1) = alpha * cos((pi * k * (2*n + 1)) / (2*N));
    end
end

% 2. Apply the DCT Matrix to the Log-Mel Energies
% Since 'filterbankEnergies' is [26 x numFrames], 
% multiplying by 'dctMatrix' [13 x 26] gives [13 x numFrames]
mfccFeatures = dctMatrix * filterbankEnergies;

% 3. Liftering (Manual)
% This smoothens the coefficients to improve recognition
lift = 22;
ceplifter = 1 + (lift/2) * sin(pi * (1:numCEPS) / lift);
% Apply liftering to each frame (each column)
for i = 1:numFrames
    mfccFeatures(:, i) = mfccFeatures(:, i) .* ceplifter';
end

% 4. Final Organization for LBG/VQ
% VQ algorithms usually expect frames as rows: [numFrames x 13]
finalFeatureMatrix = mfccFeatures';

% --- Visualization ---
figure;
imagesc(finalFeatureMatrix'); 
axis xy; colormap('jet'); colorbar;
title('Manual 13-Coefficient MFCC Features');
ylabel('Coefficient Index'); xlabel('Frame Index');

% --- MFCC Section 4: LBG (Linde-Buzo-Gray) Algorithm ---

% Parameters
data = finalFeatureMatrix; % The [Frames x 13] matrix from previous step
M = 16;                    % Desired Codebook size (must be power of 2)
epsilon = 0.01;            % Splitting parameter
max_iter = 20;             % Max iterations for K-means refinement
threshold = 0.001;         % Convergence threshold

% 1. Initialize: Start with the mean of all data (Codebook size = 1)
codebook = mean(data); 

% 2. Main LBG Loop
current_size = 1;
while current_size < M
    % --- Step A: Split the codebook ---
    % New codebook will be [v*(1+eps); v*(1-eps)]
    codebook = [codebook * (1 + epsilon); codebook * (1 - epsilon)];
    current_size = size(codebook, 1);
    
    avg_distortion = 1e10; % Initialize with a very large number
    
    % --- Step B: Iterative Refinement (K-Means) ---
    for iter = 1:max_iter
        % 1. Find Nearest Neighbor for each data point
        % We calculate Euclidean Distance manually
        num_data = size(data, 1);
        dist_matrix = zeros(num_data, current_size);
        
        for k = 1:current_size
            % Distance = sqrt(sum((data - center)^2))
            diff = data - repmat(codebook(k,:), num_data, 1);
            dist_matrix(:, k) = sum(diff.^2, 2); % Using squared distance is faster
        end
        
        [min_dist, cluster_idx] = min(dist_matrix, [], 2);
        
        % 2. Update Centroids
        new_avg_distortion = mean(min_dist);
        for k = 1:current_size
            relevant_points = data(cluster_idx == k, :);
            if ~isempty(relevant_points)
                codebook(k, :) = mean(relevant_points, 1);
            end
        end
        
        % 3. Check for convergence
        if abs(avg_distortion - new_avg_distortion) / avg_distortion < threshold
            break;
        end
        avg_distortion = new_avg_distortion;
    end
    fprintf('LBG: Codebook size %d reached.\n', current_size);
end

% --- Visualization ---
figure;
subplot(1,2,1); plot(finalFeatureMatrix(:, 1:2), '.'); title('Original MFCC Points (C1 vs C2)');
subplot(1,2,2); plot(codebook(:, 1:2), 'ro', 'LineWidth', 2); title('LBG Codebook Centers');

% --- MFCC Section 6: Database Matching ---

% 1. Get a list of all enrolled users in the current folder
dbFiles = dir('db_*.mat'); 

if isempty(dbFiles)
    error('No users found in the database! Please run enrollment first.');
end

% 2. Assume 'testFeaturesMatrix' is extracted from the current login attempt
num_test_frames = size(testFeaturesMatrix, 1);
bestDistortion = inf;
matchedUser = 'None';

fprintf('Searching database for a match...\n');

% 3. Iterate through every user in the database
for f = 1:length(dbFiles)
    % Load the codebook for the current database entry
    currentFile = dbFiles(f).name;
    dataLoad = load(currentFile);
    dbCodebook = dataLoad.codebook;
    
    % --- Calculate Distortion for this user ---
    num_cb_vectors = size(dbCodebook, 1);
    total_distortion = 0;
    
    for i = 1:num_test_frames
        current_frame = testFeaturesMatrix(i, :);
        
        % Euclidean Distance (Manual)
        diff = dbCodebook - repmat(current_frame, num_cb_vectors, 1);
        sq_dists = sum(diff.^2, 2);
        total_distortion = total_distortion + min(sq_dists);
    end
    
    avg_distortion = total_distortion / num_test_frames;
    
    % Extract name from filename for display
    [~, nameOnly, ~] = fileparts(currentFile);
    fprintf('Tested against %s | Distortion: %.4f\n', nameOnly, avg_distortion);
    
    % Check if this is the best match so far
    if avg_distortion < bestDistortion
        bestDistortion = avg_distortion;
        matchedUser = nameOnly;
    end
end

% 4. Final Decision
threshold = 5.0; % You must tune this value!



if bestDistortion < threshold
    fprintf('\n[ ACCESS GRANTED ]\n');
    fprintf('Welcome, %s! (Match Score: %.4f)\n', matchedUser, bestDistortion);
else
    fprintf('\n[ ACCESS DENIED ]\n');
    fprintf('Voice does not match any enrolled user. (Best Score: %.4f)\n', bestDistortion);
end