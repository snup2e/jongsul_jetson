function [ clust, cov_mat, mu_mat, phi] = GMM_EM( signal_param, clusters)


%% ---------- Parameter Intilization ------------------

[row col] = size(signal_param);
num_clusters = clusters;
rand_idx = randperm(row,num_clusters);

phi = (1/num_clusters)*(ones(1,num_clusters));
mu_mat =  zeros(num_clusters,size(signal_param,2));

cov_mat = cov(signal_param);


for i = 1:num_clusters
    
mu_mat(i,:) = signal_param(rand_idx(i),:);
cov_mat(:,:,i) = cov(signal_param);

end

%% Optimization through Expectation and MAximization 
iter = 1000;
err = inf;

while err > 1e-12
    %% Expectaion: 
    prvs_mu = mu_mat;
    for j = 1 : length(phi)
        w(:,j)=phi(j)*mvnpdf(signal_param,mu_mat(j,:),cov_mat(:,:,j));
    end   
w=w./repmat(sum(w,2),1,size(w,2));


w(isnan(w))=0;

%% Maximum: 
    phi = sum(w,1)./size(w,1); 
    
    mu_mat = w'*signal_param; 
    mu_mat= mu_mat./repmat((sum(w,1))',1,size(mu_mat,2));
        
    for j = 1 : length(phi)
        vari = repmat(w(:,j),1,size(signal_param,2)).*(signal_param - repmat(mu_mat(j,:),size(signal_param,1),1)); 
        cov_mat(:,:,j) = (vari'*vari)/sum(w(:,j),1) + .0001 * eye(size(signal_param,2));       
    end
    
    err = abs(norm(mu_mat,2) - norm(prvs_mu,2));
    
end

%% Estimation
[c, estimate] = max(w,[],2);


one = find(estimate==1); 
two = find(estimate == 2);




%% Returning clusters indices
clust = {find(estimate==1) , find(estimate==2)  };


end

