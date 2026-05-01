function [ Event_loc, Features] = Event_Extract( Evnt_Prdctd, signal, sigma, prsn )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

fs=8000;
tm = (0:length(signal)-1)/fs; 

new_sig = zeros(size(signal,1),1);
Event_loc = zeros(size(signal,1),1);



Detctd_Evnt_idx = find(Evnt_Prdctd(:,3) == 1);
j=1;
iter = 1;

footfall_len = 1500;
Sig = zeros(1,footfall_len);
k= 1:footfall_len;
% sigma = 2.5;
% figure()
while j < length(Detctd_Evnt_idx)
    
    fprintf('\n================= Person %d and Event %d =================\n',prsn, iter )

    stp = 0;
    while (Detctd_Evnt_idx(j+1) == Detctd_Evnt_idx(j)+1)
        j = j+ 1;
        stp = stp+1;
        if j == length(Detctd_Evnt_idx)
            break;
        end
    end
    
    Evnt_Start = Evnt_Prdctd(Detctd_Evnt_idx(j-stp),1);
    Evnt_Stop =  Evnt_Prdctd(Detctd_Evnt_idx(j),2);  
    
    [~ , mn] = max(abs(signal(Evnt_Start:Evnt_Stop)));
    mn = mn + Evnt_Start;
   wndw = tukeywin(footfall_len,0.5); % turkey window

    

    strt = mn - 400;
    stop = strt + footfall_len-1;
    if strt < 1
        strt = 1;
        wndw = wndw(1:stop-strt+1);
    end
    
    if stop >= length(signal)
        stop = length(signal);
        wndw = wndw(1:stop-strt+1);
    end
    
    w_diag = diag(wndw);
    sig = (signal(strt:stop));
    Sig(:,1:length(sig)) = w_diag*sig;

     evnt_len(iter,:) = sum(Sig ~=0);  
    

    Features(iter,:) = Sig; % zero padding in case the signals is smaller
    Event_loc(mn,:) = 0.5;


j = j + 1
iter = iter + 1

end




end

