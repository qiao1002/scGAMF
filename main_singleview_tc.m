clc
clear
% 'Grover','Tasic','Zeisel','Macosko','Treutlein','Darmanis','Ting','Pollen',
dataset = {'Treutlein','Goolam','Deng','19_Engel','Usoskin','Kold','Tasic'};
result_nmi=zeros(11,11);
result_ari=zeros(11,11);
result_nmi_ith=cell(1,12);
result_ari_ith=cell(1,12);
result_nmi_num=cell(1,12);
result_ari_num=cell(1,12);
for num =1:6
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
    miu = 0.1;
    beta=1;
    lambda1= 10^(i);
    for ith=1:5
        a = 0;
       for i = (-2):1:(-1) 
            a=a+1; 
           lambda= 10^(i);
            b=0;
            for ii = (0):0.1:(1) 
                b=b+1;
                beta= 10^(ii);
                [similarity,g,w, objValue_residual,objValue] = scGAMF_singleview(XX,X,layers,lambda,K,beta,lambda1,miu,ith);%所提模型的输出Cv作为GAE的输入亲和矩阵
                [result_label, kerNS]= SpectralClustering(similarity,num_class);  %%
                NNMI=Cal_NMI_newused(gnd, result_label);
                AARI=Contingency_ARI_newused(gnd, result_label);
                [result_nmi(a,b)]=NNMI;
                [result_ari(a,b)]=AARI;
            end
       end
       result_nmi_ith{1,ith}=result_nmi;
       result_ari_ith{1,ith}=result_ari;
    end
    result_nmi_num{1,num}=result_nmi_ith;
    result_ari_num{1,num}=result_ari_ith;
end
% ydata = tsne_bo(similarity, true_labs, num_class);
% feature_v(similarity,X,true_labs,Genes,cell_name);
