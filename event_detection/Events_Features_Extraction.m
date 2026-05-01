function [ feat ] = Events_Features_Extraction( Fs,sig )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

i=1;


feat(i) = std(sig); i = i+1;
feat(i) = kurtosis(sig); i= i+1;
% Add Mean Squared Error (MSE)                  % Calculate MSE
feat(i) = rms(sig); i = i + 1;            % Add MSE as a feature
feat(i) = quantile(sig, 0.25); i = i+1;
%feat(i) = mean(sig); i = i+1;
% feat(i)  = norm(sig,2); i = i+1;
% Compute Shannon Energy
%shannon_energy = -sum(sig.^2 .* log(sig.^2 + eps)); % eps added for numerical stability
%feat(i) = shannon_energy; i = i + 1;


L=length(sig);
NFFT = 8*2^nextpow2(L); % Next power of 2 from length of y
fft_sig = fft(sig,NFFT)/L;
fft_sig=2*abs(fft_sig(1:NFFT/2+1));
f = Fs/2*linspace(0,1,NFFT/2+1);

step = 40 ;
% feat(i)=(norm(fft_sig(1:sum(f <= 2)))^2);i=i+1; % 14

 for j = 40:step:120
 
      feat(i)=(norm( fft_sig(  sum(f <= j)+1 : sum(f <= j+step) )  )^2);i=i+1;
 
 end





end

