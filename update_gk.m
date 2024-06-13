function [D,g]=update_gk(K,C, V,layers,Q,n,beta)
%% update Multiple Core Coefficient gi and  combined kernel D
% input:
% K{v}：Kernel matrices of all intermediate embeddings for each view
% V:Number of views; layers:Number of layers of GAE;
% Q:Number of kernel functions; n:Number of cells
% output:
% D{v}:The combined kernel K of the vth view
    h=cell(V);
    r=layers*Q; % Total number of kernels for each view
    for i=1:V
        h{i}=zeros(r,1);% r--The number of kernels
    end
    for i=1:V
        for j=1:r       
            h{1,i}(j,1)=.5*trace(K{i}(:,:,j)-2*beta*K{i}(:,:,j)*C{i}+C{i}'*K{i}(:,:,j)*C{i});  
        end
    end
    for i=1:V
        for j=1:r       
            g{i}(j)=(h{1,i}(j,1)*sum(1./h{1,i}(:,1)))^(-2);  
%             g(j)=(h(j)*sum(1./h))^(-2); 
        end
    end
    D=cell(1,V);
    for i=1:V
        D{i}=zeros(n);% r--The number of kernels
    end
    for i=1:V
        for j=1:r  
             D{i}=D{i}+g{i}(j)*K{i}(:,:,j);
        end
    end
end