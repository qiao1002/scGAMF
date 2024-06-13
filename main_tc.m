clc
clear
% 'Grover','Tasic','Ting','Zeisel','Pollen','Macosko',
dataset = {'islet','Goolam','Treutlein','Deng','19_Engel','Usoskin','Kold','Tasic','Darmanis',};
result_nmi=zeros(11,11);
result_ari=zeros(11,11);
result_nmii=cell(1,12);
result_arii=cell(1,12);
for num =1:1
%%   scclrr--data preprocessing  
    load(['Data_',dataset{num},'.mat']);
    [in_X,] = FilterGenesZero(in_X);
    fea=in_X;
    [smp_num,feature_num]=size(fea);
    gnd=true_labs(:);
    num_class =length(unique(gnd)); % The number of classes
    if num_class==48%Tasic
        num_class=49;
    end
    % ---------- Initilization for X  -------- %
    fea = double(fea);
    select_sample = [];
    select_gnd    = [];
    for i = 1:num_class
        idx = find(gnd == i);%把gnd=i的索引放在idx中
        idx_sample    = fea(idx,:);%在原始矩阵中找到类别为i的所有细胞
        select_sample = [select_sample;idx_sample];
        select_gnd    = [select_gnd;gnd(idx)];
    end
    fea = select_sample';%select_sample，select_gnd已经按照类别从小到大排好序了。
    fea = fea./repmat(sqrt(sum(fea.^2)),[size(fea,1) 1]);
    gnd = select_gnd;
    X = fea;
    clear fea select_gnd select_sample idx
    
    [K, T, X1] = dataprossing(X);
    if feature_num<8000
         XX = {X1(1:50,:),X1(1:100,:),X1(1:150,:),X1(1:200,:),X1(1:250,:)};
    elseif (8000<feature_num)&&(feature_num<12000)
         XX = {X1(1:50,:),X1(1:100,:),X1(1:200,:),X1(1:400,:),X1(1:800,:)};
    else
         XX = {X1(1:200,:),X1(1:400,:),X1(1:600,:),X1(1:800,:),X1(1:1000,:)};
    end
    layers = 2;
    miu = 0.1;%isba=1
    lambda1= 10^(-1);
    a = 0;
   for i = (-2):1:(-2) 
        a=a+1;  
        lambda= 10^(i);
        b=0;
        for ii = (0.6):0.1:(0.6)  
            b=b+1;
            beta= 10^(ii);
            [similarity,g,w, objValue_residual,objValue] = scGAMF(XX,layers,lambda,K,beta,lambda1,miu);%所提模型--的输出Cv作为GAE的输入亲和矩阵
            [result_label, kerNS]= SpectralClustering(similarity,num_class);
            NNMI=Cal_NMI_newused(gnd, result_label);
            AARI=Contingency_ARI_newused(gnd, result_label);
            [result_nmi(a,b)]=NNMI;
            [result_ari(a,b)]=AARI;
        end
   end
    result_nmii{1,num}=result_nmi;
    result_arii{1,num}=result_ari;
end
% ydata = tsne_bo(similarity, true_labs, num_class);
% feature_v(similarity,X,true_labs,Genes,cell_name);
