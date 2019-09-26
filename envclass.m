function c = envclass(x,fs)
%% ENVCLASS performs the classification of an input audio signal into 1 of 5 environmental audio classes 
%Written by - Himangshu Chowdhury & Blake Diamond
%   x is the vector obtained by loading the audio file into Matlab
%   fs is the sample rate
%   c is an integer number indicating the identified environmental class
%   0 - Rain, 1 - Waves, 2 - Fire, 3 - Crickets, 4 - Birds 

%% Pre-processing 
%x = 'Waves_3.wav'
%fs = 44100;
%[xx,~] = audioread(x);
load('vars.mat','Xtrainval','Ytrainval');
features = zeros(1,8);
xx = x;
ts = 1/fs; %sampling period aka time between samples
timeBlockLength = 50 * 10^(-3); %50 ms - constant time for each block length
blockLength = floor(timeBlockLength/ts); %block length
hopLength = floor(blockLength/5); %Hop Length
totalLength = length(xx); %length of the input signal
numBlocks = ceil(totalLength/hopLength); %number of blocks
t = ((0:numBlocks-1)*hopLength + (blockLength/2))/fs; %time vector

eps = 1E-15;
kcc=2:5; 
ncc = length(kcc);
energy = zeros(numBlocks,1); %energy carried by the signal
sf = zeros(numBlocks,1); %half-rectified spectral flux of the signal
sc = zeros(numBlocks,1); %spectral centroid 
cc = zeros(numBlocks,ncc); %cepstral coefficients vector
logenergy = zeros(numBlocks,1); %log of the energy carried by the signal
cf = zeros(numBlocks,1); %center frequency
crest = zeros(numBlocks,1); %spectral crest
flatness = zeros(numBlocks,1); %spectral flatness
%% Step 1
%block based processing to extract summary features from x
   %% Finding Features
    for n = 1:numBlocks
        %% calculate start/stop indexes
        iStart = (n-1)*hopLength + 1;
        iStop = min(totalLength,iStart + blockLength -1);
        len = iStop - iStart + 1;
        sig = xx(iStart:iStop);
        %% Find several features of the signal - look at Lecture 10 slides
        %energy
        energy(n) = mean(sig.^2 .* hann(len));
        
        %log energy
        prelogEnergy = sig.^2 .* hann(len);
        logenergy(n) = mean(log(prelogEnergy + 0.00001));
        
        %cepstral coefficients
        f = fft(sig .* hann(len));
        f = real(log(f));
        f = real(ifft(f));
        cc(n,:) = f(kcc); %each iteration, length 4 vector of cepstral coeff created
        
        %avg specteral centroid
        sumF = 0;
        F = fft(sig .* hann(len), 2*len);
        for k = 1:length(F)
            sc(n) = sc(n) + k*abs(F(k));
            sumF = sumF + abs(F(k));
        end
        sc(n) = sc(n)/sumF;
        
        %spectral flux
        NewDFT = fft(sig .* hann(len), 2*len);
        if n == 1
            OldDFT = zeros(length(NewDFT),1);
        end
        if length(NewDFT) ~= length(OldDFT)
            NewDFT = [NewDFT; zeros(length(OldDFT)-length(NewDFT),1)];
        end
        sf(n) = sqrt(sum((abs(NewDFT)-abs(OldDFT)).^2));
        sf(n) = (sf(n)+abs(sf(n)))/2;
        OldDFT = NewDFT;
        
        %center frequency
        cf(n) = findCF(xx,fs,iStart,iStop);
        
        q = fft(sig .* hann(len)); %this was called f, but I changed it to q so that I could use it below, from cepstral coeff
        %spectral crest
        crest(n) = 10* log10(max(q)/mean(q));
       
        %spectral flatness
        flatness(n,:) = exp(mean(log(q)))./ (mean(q)+eps);
       
        %entropy
        specN = sig/sum(sig);
        entropy(n) = -real(sum((specN .* log(specN+eps))./log(q))/length(q)); %average entropy over the whole spectrum
    end

%% Step 2
% determine which summary features are best for clasiffying sounds
% plot each class as a different color
% see lecture 12 for an example

% assignment for training
%features(1) = mean(sc);
features(1) = mean(sf);
features(2:5) = mean(cc);
features(6) = std(energy);
features(7) = mean(cf);
% features(8) = mean(crest); %measures ratio of peaks to average value
% features(9) = mean(flatness);
% features(10) = std(entropy);
%may need to change above indexes for output vector
% Plotting for visualization
% figure(2)
% subplot(5,1,1); plot(xx/max(xx),'-b'); title(['Original Time Domain - ',x]);
% 
% subplot(5,1,2); 
% plot(t',cc(:,1),'-b'); hold on 
% plot(t',cc(:,2),'-m'); hold on
% plot(t',cc(:,3),'-r'); hold on
% plot(t',cc(:,4),'-g'); hold off
% title('Cepstral Coefficients');
% legend('1','2','3','4');
% 
% subplot(5,1,3); plot(t',energy/max(energy)); title('Energy');
% 
% subplot(5,1,4); plot(t',rms_specflux/max(rms_specflux),'-b'); title('Rectified Spectral Flux');
% 
% subplot(5,1,5); plot(t',sc,'-b'); title('Average Spectral Centroid - not normalized');

%% Step 3
% determine the class c of the signal based on the summary features using a
% classifier
% user k-Nearest neighbors approach as described in L12

c = KNN(9,Xtrainval,Ytrainval,features,'Euclidian');

function CF = findCF(xx,FS,first,last)
% xx is an input 
% FS is an input
% CF, which is the output, is the center frequency
warning('off','signal:findpeaks:largeMinPeakHeight')
temp_signal = xx(first:last);
offset = last - first + 1;
%% find frequency spectrum
signal_fft = abs(fft(temp_signal,6*offset)); %zero padding
signal_fft = signal_fft(1:round(length(signal_fft)/2)); %look only at positive frequencies
blockMax = max(signal_fft); %highest amplitude within frequency block
%create local frequency vector
fftlength = length(signal_fft); 
stepsize = (FS/2)/fftlength; %fiding step size 
fBlock = 0:stepsize:FS/2; %creating vector that goes from zero to highest possible frequency
fBlock = fBlock(1:fftlength)';%adjust size of vector

%% Find maximum amplitude and frequencies
[maxPks, maxLocs] = findpeaks(signal_fft/(max(signal_fft)),fBlock,...
    'SortStr','descend',...
    'NPeaks',3,...     
    'MinPeakDistance',100,...
    'MinPeakHeight',0.17);

%% Generate sine waves and store 
if (length(maxPks)==3) %three peaks are found
    
    %calculate fundamental frequency & amplitude
    maxLocsSort= sort(maxLocs); 
    fundFreq = min(abs(maxLocsSort(2)-maxLocsSort(1)),abs(maxLocsSort(2)-maxLocsSort(3)));
elseif (length(maxPks)==2) %two peaks are found 
    
    %calculate fundamental frequency & amplitude
    maxLocsSort= sort(maxLocs);
    fundFreq = min(abs(maxLocsSort(2)-maxLocsSort(1)),maxLocsSort(1)); %minimum between difference and lowest frequence
elseif (isempty(maxPks))%no peaks are found
    fundFreq = 0;
else
    %calculate fundamental frequency & amplitude
    maxLocsSort= sort(maxLocs);
    fundFreq = maxLocsSort(1);
end
    CF = fundFreq;
end

function predicted_labels = KNN(k,train_data,train_labels,test_data,distmethod)
%KNN_: classifying using k-nearest neighbors algorithm

%checks
if nargin < 4
    error('Too few input arguments.')
elseif nargin < 5
    distmethod = 'Euclidian';
% elseif nargin < 6 && distmethod == 'Minkowski'
%     p = 3;
end
if size(train_data,2) ~= size(test_data,2)
    error('data should have the same dimensionality');
end
%initialization
predicted_labels = zeros(size(test_data,1),1);
ed = zeros(size(test_data,1),size(train_data,1)); %ed: (MxN) euclidean distances 
sqed = zeros(size(test_data,1),size(train_data,1)); %sqed: (MxN) squared euclidian distances
man = zeros(size(test_data,1),size(train_data,1)); %man: (MxN) manhattan distances
mink = zeros(size(test_data,1),size(train_data,1)); %mink: (MxN) minkowski distances
mah = zeros(size(test_data,1),size(train_data,1)); %mah: (MxN) mahalanobis distances
ind = zeros(size(test_data,1),size(train_data,1)); %corresponding indices (MxN)
k_nn = zeros(size(test_data,1),k); %k-nearest neighbors for testing sample (Mxk)
switch distmethod
    case 'Euclidian'
        %calc euclidean distances between each testing data point and the training data samples
        for test_sample = 1:size(test_data,1)
            for train_sample = 1:size(train_data,1)
                %calc and store sorted euclidean distances with corresponding indices
                ed(test_sample,train_sample) = sqrt(sum((test_data(test_sample,:) - train_data(train_sample,:)).^2));
            end
            [ed(test_sample,:),ind(test_sample,:)] = sort(ed(test_sample,:));
        end
    case 'SquareEuclidian'
        %calc square euclidean distances between each testing data point and the training data samples
        for test_sample = 1:size(test_data,1)
            for train_sample = 1:size(train_data,1)
                %calc and store sorted square euclidean distances with corresponding indices
                sqed(test_sample,train_sample) = sum((test_data(test_sample,:) - train_data(train_sample,:)).^2);
            end
            [sqed(test_sample,:),ind(test_sample,:)] = sort(sqed(test_sample,:));
        end
    case 'Manhattan'
        %calc manhattan distances between each testing data point and the training data samples
        for test_sample = 1:size(test_data,1)
            for train_sample = 1:size(train_data,1)
                %calc and store sorted manhattan distances with corresponding indices
                man(test_sample,train_sample) = sum(abs(test_data(test_sample,:) - train_data(train_sample,:)));
            end
            [man(test_sample,:),ind(test_sample,:)] = sort(man(test_sample,:));
        end
    case 'Minkowski'
        %calc minkowski distances between each testing data point and the training data samples
        p = 4;
        for test_sample = 1:size(test_data,1)
            for train_sample = 1:size(train_data,1)
                %calc and store sorted minkowski distances with corresponding indices
                mink(test_sample,train_sample) = nthroot(sum(abs((test_data(test_sample,:) - train_data(train_sample,:))).^p),p);
            end
        [mink(test_sample,:),ind(test_sample,:)] = sort(mink(test_sample,:));
        end
    case 'Mahalanobis'
        %calc mahalanobis distances between each testing data point and the training data samples
        for test_sample = 1:size(test_data,1)
            for train_sample = 1:size(train_data,1)
                %calc and store sorted mahalanobis distances with corresponding indices
                mah(test_sample,train_sample) = sqrt(sum((test_data(test_sample,:) - train_data(train_sample,:))*(cov(test_data(test_sample,:),train_data(train_sample,:)))'*(test_data(test_sample,:) - train_data(train_sample,:))'));
            end
        [mah(test_sample,:),ind(test_sample,:)] = sort(mah(test_sample,:));
        end
    otherwise
        %incorrect input
        disp('Choose one of the following distance calculation methods: Euclidian, SquareEuclidian, Manhattan, Minkowski, Mahalanobis')
end

%find the nearest k for each data point of the testing data
k_nn = ind(:,1:k);
%get the majority vote 
for i = 1:size(k_nn,1)
    options = unique(train_labels(k_nn(i,:)'));
    max_count = 0;
    max_label = 0;
    for j = 1:length(options)
        L = length(find(train_labels(k_nn(i,:)') == options(j)));
        if L > max_count
            max_label = options(j);
            max_count = L;
        end
    end
    predicted_labels(i) = max_label;
end

end


end
