function [S,C,g,w,objValue_residual,objValue]=KSLS(XX,V,layers, smp_num,lambda,beta,miu)
% 具有自适应相似性保留的多核自表达学习
    n=smp_num;
    max_mu = 10^4;  
    rho = 1.5;%rho = 1.2;
     zr = 1;
    islocal = 1; 
    ker=5;
    %% Construct the kernel matrix K(v=1,...,V) of each view; Each Kv contains layers*5 different kernel matrices
    ker_num = layers*ker;%ker_ num is the total number of kernel matrices for each view;5 kernel functions
    K = cell(V,1);
    for v=1:V
        K{v} = zeros(n, n, ker_num);%每个Kv是一个三元组
    end
    for v=1:V
        for i=1:layers
            options.KernelType = 'Gaussian';
            K{v}(:,:,1+(i-1)*5) = construct_kernel(XX{v}{i}', [], options);
            options.KernelType = 'Polynomial';
            options.d = 3;
            K{v}(:,:,2+(i-1)*5) = construct_kernel(XX{v}{i}', [], options);
            options.KernelType = 'Linear';
            K{v}(:,:,3+(i-1)*5) = construct_kernel(XX{v}{i}', [], options);
            options.KernelType = 'Sigmoid';
            options.c = 0;
            options.d = 0.1;
            K{v}(:,:,4+(i-1)*5) = construct_kernel(XX{v}{i}', [], options);
            options.KernelType = 'InvPloyPlus';
            options.c = 0.01;
            options.d = 1;
            K{v}(:,:,5+(i-1)*5) = construct_kernel(XX{v}{i}', [], options);
        end
    end
    for v=1:V
        K{v} = kcenter(K{v});
        K{v} = knorm(K{v});
    end
    %% initialize  Cv(使用KNN用Hv初始化每个Cv),S(通过ωv对{Cv}取平均值初始化S)
    C = cell(1,V);
    for i = 1:V
        for j=1:layers
            [C{i}, ~] = InitializeSIGs(XX{i}{j}, 10, 0);
%             [C{i}]= simlarityGS(XX{i}{j});
        end
    end
    C0 = C;
    S = zeros(n);
    for i = 1:V
        S = S + C0{i};
    end
    S = S/V;
    for j = 1:V
        S(j,:) = S(j,:)/sum(S(j,:));
    end
    %% initialize Jv=0,Zv=0,Y1v=0,Y2v=0,combined kernel D of the v-th view
    J = cell(1,V);Y1 = cell(1,V);Z = cell(1,V);D= cell(1,V);dist = cell(1,V);
    for i=1:V
        J{1,i}=zeros(n);
        J0{1,i}=zeros(n);
        Z{1,i}=zeros(n);
        Y1{1,i}=zeros(n);
        D{1,i}=zeros(n);
        dist{1,i}=zeros(n);%combined kernel of the v-th view
    end
    w = ones(1,V)/V;%Consensus learning coefficient
   %% initialize g_v=1/ker_num, combined kernel Dv
   for i=1:V
        g{i} = ones(1,ker_num)/ker_num;
        for j = 1:ker_num
            D{i} = D{i} + g{i}(j)*K{i}(:,:,j);
        end
   end
   T1=cell(V);T2=cell(V);
  
   for iter=1:20 
       
       %% Update Jv--Y1 F范数
%         S1 = S;
%         for i = 1: V%
%             J{i} = (2*lambda*eye(n)+miu*eye(n))\(miu*C{i}-Y1{i}); 
%         end
       %%  Update Jv--Y1 核范数
        for i = 1: V
            [AU,SU,VU] = svd(C{i}+Y1{i}/miu,'econ');
            AU(isnan(AU)) = 0;
            VU(isnan(VU)) = 0;
            SU(isnan(SU)) = 0;
            SU = diag(SU);
            SVP = length(find(SU>lambda/miu));
            if SVP >= 1
                SU = SU(1:SVP)-lambda/miu;
            else
                SVP = 1;
                SU = 0;
            end
            J{i} = AU(:,1:SVP)*diag(SU)*VU(:,1:SVP)';
        end
       %%  Update Jv--Y1 L1范数
%         for i = 1: V
%             temp_J = C{i} + Y1{i}/miu;
%             J{i} = max(C{i} + Y1{i}/miu - lambda/miu, 0) + min(temp_J + lambda/miu, 0);
%             J{i} = max(J{i},0);
%         end
        %% Update Cv
        for i = 1: V
            T1{1,i}=(D{i}')+D{i};
            T2{1,i}=miu*(J{i})-Y1{i}+2*w(v)*S;
            C{i}=(0.5*T1{1,i}+miu*eye(n)+2*w(v)*eye(n))\(beta*D{i}'+T2{1,i}); 
            C{i} = C{i}- diag(diag(C{i}));
            C{i}(find(C{i}<0))=0;
            C{i}(isnan(C{i})) = 0;           
        end
        %% Update S
        S1 = zeros(n);
        S1 = S;
        U = zeros(n);
        for i = 1 : n
            idx = zeros();
            for v = 1 : V
                s0 = C{v}(i,:);
                idx = [idx,find(s0>0)];
            end
            idxs = unique(idx(2:end));
            if islocal == 1
                idxs0 = idxs;
            else
                idxs0 = 1:n;
            end
            sumSJ = zeros(1, length(idxs0));
            for v = 1:V
                s1 = C{v}(i,:);
                si = w(v) .* s1(idxs0);
                sumSJ = sumSJ + si;
            end
            U(i,idxs0) = EProjSimplex_new(sumSJ / sum(w));
        end
       S=U;
       %% Update W
       w1 = ones(1,V)/V;
       w1 = w;
        for v = 1:V
            US = S - C{v};
            distUS = norm(US, 'fro')^2;
            if distUS == 0
                distUS = eps;
            end
            w(v) = 0.5/sqrt(distUS);
        end
        %% update 多核系数gi and combined kernel Dv
        [D,g]=update_gk(K,C,V,layers,ker,n,beta);
        % -------- Update Y1 Y2 miu -------- %
        for i=1:V
            Y1{i} = Y1{i} + miu*(C{i}-J{i});
        end
        miu = min(max_mu,rho*miu);
        % --------- obj ---------- %
        curValue_J=0;
        curValue_Z=0;
        curValue_S=0;
        curValue_W=0;
        curValue=0;
        for v = 1 : V
          curValue_J =  curValue_J+(C{v}-J{v});
        end
        curValue_J=curValue_J/V;
        curValue_S=curValue_S+(S-S1);
         curValue_W=curValue_W+(w-w1);
        curValue=curValue/V;
         objValue_residual(iter) = max(max(abs(curValue_J(:))));
%         objValue_residual(iter) = max(max(abs(curValue_J(:))))+max(max(abs(curValue_S(:))))+max(max(abs(curValue_W(:))));
        objValue(iter) = curValue;
        if(iter > 1)
           diff = abs(abs(objValue_residual(iter)) - abs(objValue_residual(iter - 1)));
           
           if diff <= zr  
               fprintf('The algorithm has reached convergence at the %d-th iteration\n', iter);
               break;
           end
        end    
    end
end

    %% 
    % actual_ids= kmeans(F, c, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');
    % [result] = ClusteringMeasure( actual_ids,s);
    function d = L2_distance_1(a,b)
    % compute squared Euclidean distance欧氏距离的平方
    % ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
    % a,b: two matrices. each column is a data
    % d:   distance matrix of a and b
    if (size(a,1) == 1)
        a = [a; zeros(1,size(a,2))];
        b = [b; zeros(1,size(b,2))];
    end
    aa=sum(a.*a); bb=sum(b.*b); ab=a'*b;
    d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;
    d = real(d);
    d = max(d,0);
    % % force 0 on the diagonal?
    % if (df==1)
    %   d = d.*(1-eye(size(d)));
    % end
    end
    
    function result=cosSim(data)
    %COSSIM Summary of this function goes here余弦相似度
    %Detailed explanation goes here
    rows=size(data,1);
    result=zeros(rows,rows);
    for i=1:rows
        for j=1:i
            if (norm(data(i,:))*norm(data(j,:))==0)
                result(i,j)=0;
            else
                result(i,j)=dot(data(i,:),data(j,:))/(norm(data(i,:))*norm(data(j,:))); 
            end
            result(j,i)=result(i,j);
        end
    end
    end
    function [x ft] = EProjSimplex_new(v, k)
    %% Optimization Problem
    %
    %  min  1/2 || x - v||^2
    %  s.t. x>=0, 1'x=k
    %
    if nargin < 2
        k = 1;
    end
    ft=1;
    n = length(v);
    v0 = v-mean(v) + k/n;
    %vmax = max(v0);
    vmin = min(v0);
    if vmin < 0
        f = 1;
        lambda_m = 0;
        while abs(f) > 10^-10
            v1 = v0 - lambda_m;
            posidx = v1>0;
            npos = sum(posidx);
            g = -npos;
            f = sum(v1(posidx)) - k;
            lambda_m = lambda_m - f/g;
            ft=ft+1;
            if ft > 100
                x = max(v1,0);
                break;
            end
        end
        x = max(v1,0);
    else
        x = v0;
    end
    end