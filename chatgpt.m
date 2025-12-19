function chatgpt()
    clc; close all;

    disp('==================================================');
    disp('   Text-Dependent Voice Password System (BUET)    ');
    disp('==================================================');
    disp('1. Enroll a New User');
    disp('2. Verify a User');
    disp('3. Exit');
    disp('--------------------------------------------------');

    choice = input('Enter choice (1/2/3): ');
    if choice == 3
        return;
    end

    if choice == 1
        defaultFile = 'train.wav';
        fprintf('\n--- ENROLLMENT MODE ---\n');
    else
        defaultFile = 'test.wav';
        fprintf('\n--- VERIFICATION MODE ---\n');
    end

    filename = input(['Enter filename (Enter = ' defaultFile '): '],'s');
    if isempty(filename), filename = defaultFile; end
    if ~isfile(filename), error('Audio file not found.'); end

    fprintf('Extracting MFCC features...\n');
    features = extract_mfcc_features(filename);

    % ================= ENROLL =================
    if choice == 1
        user = input('Enter User ID: ','s');
        if isempty(user), user = 'User1'; end

        codebook = run_lbg(features);
        save(['db_' user '.mat'],'codebook');

        fprintf('User "%s" enrolled successfully.\n',user);

    % ================= VERIFY =================
    else
        db = dir('db_*.mat');
        if isempty(db), error('No enrolled users found.'); end

        bestDist = inf;
        bestUser = 'None';

        fprintf('\n%-15s | %-10s\n','User','Distortion');
        fprintf('-------------------------------\n');

        for k = 1:length(db)
            d = load(db(k).name);
            D = vq_distortion(features,d.codebook);

            name = erase(db(k).name,{'db_','.mat'});
            fprintf('%-15s | %.4f\n',name,D);

            if D < bestDist
                bestDist = D;
                bestUser = name;
            end
        end

        THRESHOLD = 120;   % tune experimentally

        fprintf('-------------------------------\n');
        if bestDist < THRESHOLD
            fprintf('ACCESS GRANTED → %s\n',bestUser);
        else
            fprintf('ACCESS DENIED\n');
        end
    end
end

% =====================================================================
% MFCC EXTRACTION (NO TOOLBOX)
% =====================================================================
function features = extract_mfcc_features(filename)

    [x,fs] = audioread(filename);
    x = mean(x,2);                 % mono
    x = x - mean(x);               % DC removal

    % Pre-emphasis
    x = filter([1 -0.97],1,x);

    % Normalize
    m = max(abs(x));
    if m < 1e-6, error('Silent audio'); end
    x = x/m;

    % Simple VAD
    idx = find(abs(x) > 0.05);
    x = x(idx(1):idx(end));

    % Framing
    frameLen = round(0.025*fs);
    hop = round(0.010*fs);
    if length(x) < frameLen, error('Audio too short'); end

    numFrames = floor((length(x)-frameLen)/hop)+1;
    frames = zeros(frameLen,numFrames);
    for i=1:numFrames
        s = (i-1)*hop+1;
        frames(:,i) = x(s:s+frameLen-1);
    end

    % Hamming window
    n = (0:frameLen-1)';
    win = 0.54 - 0.46*cos(2*pi*n/(frameLen-1));
    frames = frames .* win;

    % FFT
    NFFT = 512;
    mag = abs(fft(frames,NFFT));
    pow = (mag(1:NFFT/2+1,:).^2)/NFFT;

    % Mel filterbank
    nfilt = 26;
    mel = @(f) 2595*log10(1+f/700);
    imel = @(m) 700*(10.^(m/2595)-1);

    melPts = linspace(mel(300),mel(fs/2),nfilt+2);
    hzPts = imel(melPts);
    bins = floor((NFFT+1)*hzPts/fs);
    bins(bins>NFFT/2)=NFFT/2;

    fbank = zeros(nfilt,NFFT/2+1);
    for m=2:nfilt+1
        for k=bins(m-1):bins(m)
            fbank(m-1,k+1) = (k-bins(m-1))/(bins(m)-bins(m-1)+eps);
        end
        for k=bins(m):bins(m+1)
            fbank(m-1,k+1) = (bins(m+1)-k)/(bins(m+1)-bins(m)+eps);
        end
    end

    energies = log(fbank*pow + 1e-10);

    % ===== Manual DCT =====
    numCeps = 13;
    dctM = zeros(numCeps,nfilt);
    for k=0:numCeps-1
        dctM(k+1,:) = cos(pi*k*(2*(0:nfilt-1)+1)/(2*nfilt));
    end
    mfcc = dctM * energies;

    mfcc = mfcc - mean(mfcc,2);   % CMN
    features = mfcc';
end

% =====================================================================
% LBG VECTOR QUANTIZATION
% =====================================================================
function codebook = run_lbg(data)

    M = 16;
    eps = 0.01;
    codebook = mean(data,1);

    while size(codebook,1) < M
        codebook = [codebook*(1+eps); codebook*(1-eps)];

        for it=1:20
            idx = zeros(size(data,1),1);
            for i=1:size(data,1)
                dmin = inf;
                for k=1:size(codebook,1)
                    d = sum((data(i,:)-codebook(k,:)).^2);
                    if d < dmin
                        dmin = d;
                        idx(i)=k;
                    end
                end
            end

            old = codebook;
            for k=1:size(codebook,1)
                pts = data(idx==k,:);
                if ~isempty(pts)
                    codebook(k,:) = mean(pts,1);
                end
            end

            if norm(codebook-old,'fro') < 1e-4
                break;
            end
        end
    end
end

% =====================================================================
% VQ DISTORTION (NO pdist2)
% =====================================================================
function D = vq_distortion(data,codebook)

    total = 0;
    for i=1:size(data,1)
        dmin = inf;
        for k=1:size(codebook,1)
            d = sum((data(i,:)-codebook(k,:)).^2);
            if d < dmin
                dmin = d;
            end
        end
        total = total + dmin;
    end
    D = total / size(data,1);
end
