function [similarity,g,w,objValue_residual,objValue] = scGAMF_singleview(X,X_raw,layers,lambda,k,beta,lambda1,miu,ith)
% Take the output Cv of the proposed model as the input affinity matrix of GAE
% LRMKALS_new4--多核自适应局部结构和自适应相似性LRMKALS
% LRMKLS_new4--LRMKLS(多核局部结构和自适应相似性LRMKALS)  lambda*||J||_+distX_ij*z_ij+0.5*Tr(K+Z_T*K*Z)-beta*Tr(KZ)+0.5*miu*||Z-S+C1/miu||_F2+0.5*miu*||Z-J+C2/miu||_F2
zr = 10e-11;  
% v = length(X); % 个矩阵
if ith==0
    X{1}=X_raw;
else
    X{1}=X{ith};
end
v=1;
S = cell(v, 1);
%% Data Normalization
for i = 1 : v
    % X{i} = NormalizeFea(X{i}, 0); % unit norm
    % Initialize each view
    [S{i}, ~] = InitializeSIGs(X{i}, k);
end
ZZ = cell(v, 1);
SS = cell(1, 1);
n = size(X{1}, 2); % X{j} is d*n; 细胞数
NITER = 2;
for i = 1:NITER
    if i==1
        for j = 1 : v
    %        n = size(X{j}, 2); % X{j} is d*n; 细胞数
            A_bar = S{j} + speye(n); % 一个主对角线元素为 1 且其他位置元素为 0 的 n×n 稀疏单位矩阵
            d = sum(A_bar);
            d_sqrt = 1.0./sqrt(d);
            d_sqrt(d_sqrt == Inf) = 0;
            DH = diag(d_sqrt);
            DH = sparse(DH);
            A_n = DH * sparse(A_bar) * DH;   %A_n相当于原文中的U~:d*d
%             SSS{j}=double(A_n);
            fprintf('compute A_n finished for the %-th view', j);
            gcn = A_n * X{j}';   % n*n * n*d
            [~, m] = size(gcn);%m：features 数目
            [allhx,allhx1,~] = GAE(X{j}, layers, A_n, lambda);%allhx（1*3cell）存储了编码器所有的中间嵌入
            %% 利用所用中间嵌入
            Z0 = allhx;%取allhx的后
            ZZ{j} = Z0;%ZZ:(v*1)cell,(1*layers) in each cell
            %% 只输出最后一层
            Z1 = allhx1(end - m + 1 : end, :);
            ZZ1{j} = Z1;
        end
    else
%%    Use the output Cv of the proposed model as the input to the GAE  
        for j = 1 : v
            A_bar = C{j} + speye(n); % 一个主对角线元素为 1 且其他位置元素为 0 的 n×n 稀疏单位矩阵
            d = sum(A_bar);
            d_sqrt = 1.0./sqrt(d);
            d_sqrt(d_sqrt == Inf) = 0;
            DH = diag(d_sqrt);
            DH = sparse(DH);
            A_n = DH * sparse(A_bar) * DH;   %A_n相当于原文中的U~:d*d
%             SSS{j}=double(A_n);
            fprintf('compute A_n finished for the %-th view', j);
            gcn = A_n * X{j}';   % n*n * n*d
            [~, m] = size(gcn);%m：features 数目
            [allhx,allhx1,~] = GAE(X{j}, layers, A_n, lambda);%allhx（1*3cell）存储了编码器所有的中间嵌入
            %% 利用所用中间嵌入
            Z0 = allhx;%取allhx的后
            ZZ{j} = Z0;%ZZ:(v*1)cell,(1*layers) in each cell
            %% 只输出最后一层
            Z1 = allhx1(end - m + 1 : end, :);
            ZZ1{j} = Z1;
        end
    end 
     CC=ZZ;  % 所用中间嵌入
     [similarity,C,g,w,objValue_residual,objValue]=KSLS(CC,v,layers,n, lambda1,beta,miu); % 具有自适应相似性保留的多核自表达学习
 end 
end

function idx = computeLabels(T, k)
Z = (abs(T) + abs(T')) / 2;
idx = clu_ncut(Z, k);
end