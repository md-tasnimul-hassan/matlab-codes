function voice_password_system()
    clc; close all;
    disp('==================================================');
    disp('   Text-Dependent Voice Password System (BUET)    ');
    disp('==================================================');
    disp('1. Enroll a New User (Training)');
    disp('2. Verify a User (Testing)');
    disp('3. Exit');
    disp('--------------------------------------------------');
    
    choice = input('Enter your choice (1/2/3): ');
    
    if choice == 3
        return;
    end
    
    % --- 1. FILE SELECTION ---
    if choice == 1
        defaultFile = 'train.wav';
        fprintf('\n--- ENROLLMENT MODE ---\n');
    else
        defaultFile = 'test.wav';
        fprintf('\n--- VERIFICATION MODE ---\n');
    end
    
    filename = input(['Enter filename (press Enter for "' defaultFile '"): '], 's');
    if isempty(filename)
        filename = defaultFile;
    end
    
    if ~isfile(filename)
        error('File "%s" not found! Please record it first.', filename);
    end
    
    % --- 2. PRE-PROCESSING & MFCC EXTRACTION (Common to both modes) ---
    fprintf('Processing audio "%s"... ', filename);
    features = extract_mfcc_features(filename);
    fprintf('Done.\n');
    
    % --- 3. MODE SPECIFIC LOGIC ---
    if choice == 1
        % === ENROLLMENT (LBG + SAVE) ===
        userName = input('Enter Name/ID for this user: ', 's');
        if isempty(userName), userName = 'User1'; end
        
        fprintf('Generating Codebook using LBG Algorithm...\n');
        codebook = run_lbg(features);
        
        saveName = sprintf('db_%s.mat', userName);
        save(saveName, 'codebook');
        
        fprintf('SUCCESS: User "%s" saved to database (%s).\n', userName, saveName);
        
        % Visualize
        figure;
        plot(features(:, 1), features(:, 2), '.', 'Color', [0.7 0.7 0.7]); hold on;
        plot(codebook(:, 1), codebook(:, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
        title(['Codebook for ' userName]);
        legend('MFCC Features', 'LBG Centroids');
        grid on;
        
    elseif choice == 2
        % === VERIFICATION (DATABASE MATCHING) ===
        dbFiles = dir('db_*.mat');
        if isempty(dbFiles)
            fprintf('\nERROR: No database found. Please enroll a user first.\n');
            return;
        end
        
        bestDist = inf;
        bestUser = 'None';
        
        fprintf('\nComparing against database:\n');
        fprintf('%-20s | %-10s\n', 'User', 'Distortion');
        fprintf('-----------------------------------\n');
        
        numTestFrames = size(features, 1);
        
        for k = 1:length(dbFiles)
            % Load Codebook
            data = load(dbFiles(k).name);
            cb = data.codebook;
            numCB = size(cb, 1);
            
            % Calculate Average Distortion (Vector Quantization)
            totalDist = 0;
            for i = 1:numTestFrames
                % Distance from current test frame to closest centroid
                currentFrame = features(i, :);
                diffs = cb - repmat(currentFrame, numCB, 1);
                sqDists = sum(diffs.^2, 2);
                totalDist = totalDist + min(sqDists);
            end
            avgDist = totalDist / numTestFrames;
            
            % Extract Name
            [~, name, ~] = fileparts(dbFiles(k).name);
            name = strrep(name, 'db_', ''); % Remove 'db_' prefix
            
            fprintf('%-20s | %.4f\n', name, avgDist);
            
            if avgDist < bestDist
                bestDist = avgDist;
                bestUser = name;
            end
        end
        
        % DECISION
        THRESHOLD = 10000; % <--- TUNE THIS VALUE FOR YOUR PROJECT
        fprintf('-----------------------------------\n');
        if bestDist < THRESHOLD
            fprintf('RESULT: ACCESS GRANTED to %s\n', bestUser);
        else
            fprintf('RESULT: ACCESS DENIED (Best match: %s, but score %.2f > %.2f)\n', ...
                bestUser, bestDist, THRESHOLD);
        end
    end
end

% =========================================================================
%  HELPER FUNCTION: MFCC EXTRACTION (The Engine)
% =========================================================================
function finalFeatureMatrix = extract_mfcc_features(filename)
    [signal, fs] = audioread(filename);
    if isrow(signal), signal = signal'; end
    
    % A. Pre-emphasis
    signal = filter([1 -0.97], 1, signal);
    
    % B. Safe Bandpass Filtering
    N = length(signal);
    f = fft(signal);
    freqs = (0:N-1)*fs/N;
    mask = (freqs >= 300 & freqs <= 3400) | (freqs >= fs-3400 & freqs <= fs-300);
    smooth_mask = conv(double(mask'), ones(50,1)/50, 'same');
    signal = real(ifft(f .* smooth_mask));
    
    % C. Normalization & VAD (Simple Energy Gate)
    signal = signal / max(abs(signal));
    
    % remove silence
    % Find first and last point where signal exceeds threshold
    threshold = 0.05; % Slightly higher threshold
    indices = find(abs(signal) > threshold);
    
    if ~isempty(indices)
        % Keep everything in between the first and last sound
        first_idx = indices(1);
        last_idx = indices(end);
        signal = signal(first_idx:last_idx);
    else
        % Fallback if file is basically silent
        warning('Signal is too quiet!');
    end
    
    % 1. Framing
    frameDur = 0.025; hopDur = 0.010;
    L = round(frameDur * fs); S = round(hopDur * fs);
    numFrames = floor((length(signal) - L) / S) + 1;
    frames = zeros(L, numFrames);
    for i = 1:numFrames
        idx = (i-1)*S + 1;
        frames(:, i) = signal(idx : idx + L - 1);
    end
    
    % 2. Windowing
    win = 0.54 - 0.46 * cos(2*pi*(0:L-1)'/(L-1));
    frames = frames .* win;
    
    % 3. FFT & Power Spectrum
    NFFT = 512;
    magSpec = abs(fft(frames, NFFT));
    powSpec = (magSpec(1:NFFT/2+1, :).^2) / NFFT;
    
    % 4. Mel Filterbank
    numFilters = 26;
    lowHz = 300; highHz = fs/2;
    melPoints = linspace(2595*log10(1+lowHz/700), 2595*log10(1+highHz/700), numFilters+2);
    hzPoints = 700*(10.^(melPoints/2595)-1);
    binPoints = floor((NFFT+1)*hzPoints/fs);
    
    fbank = zeros(numFilters, NFFT/2+1);
    for m = 2:numFilters+1
        for k = binPoints(m-1):binPoints(m)
            fbank(m-1, k+1) = (k - binPoints(m-1))/(binPoints(m)-binPoints(m-1));
        end
        for k = binPoints(m):binPoints(m+1)
            fbank(m-1, k+1) = (binPoints(m+1) - k)/(binPoints(m+1)-binPoints(m));
        end
    end
    
    energies = log(fbank * powSpec + 1e-10);
    
    % 5. DCT & Liftering
    numCEPS = 13;
    % Manual DCT Matrix
    dctM = zeros(numCEPS, numFilters);
    for k = 0:numCEPS-1
        alpha = sqrt((1+(k>0))/numFilters); % 1/sqrt(N) for k=0, sqrt(2/N) for k>0
        dctM(k+1, :) = alpha * cos(pi*k*(2*(0:numFilters-1)+1)/(2*numFilters));
    end
    mfcc = dctM * energies;
    
    % Liftering
    lift = 22;
    lifter = 1 + (lift/2)*sin(pi*(1:numCEPS)'/lift);
    mfcc = mfcc .* lifter;
    mfcc = mfcc - mean(mfcc, 2);
    finalFeatureMatrix = mfcc'; % Return as [Frames x 13]
end

% =========================================================================
%  HELPER FUNCTION: LBG ALGORITHM (The Trainer)
% =========================================================================
function codebook = run_lbg(data)
    M = 16; % Target size
    epsilon = 0.01;
    codebook = mean(data);
    
    while size(codebook, 1) < M
        codebook = [codebook*(1+epsilon); codebook*(1-epsilon)];
        % K-Means Refinement
        for i = 1:20
            numCB = size(codebook, 1);
            dists = zeros(size(data,1), numCB);
            for k = 1:numCB
                dists(:,k) = sum((data - codebook(k,:)).^2, 2);
            end
            [~, idx] = min(dists, [], 2);
            
            % Update centers
            oldCB = codebook;
            for k = 1:numCB
                pts = data(idx==k, :);
                if ~isempty(pts), codebook(k,:) = mean(pts); end
            end
            
            if norm(codebook-oldCB) < 1e-4, break; end
        end
    end
end