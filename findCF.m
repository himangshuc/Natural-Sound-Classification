function CF = findCF(xx,FS,first,last)
% xx is an input 
% FS is an input
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
    