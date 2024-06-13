function [K, T, X1] = dataprossing(X)
[m,n] = size(X); % m个特征，n个细胞
VAR = var(X,0,2);
VAR = VAR';
[VAR, index] = sort(VAR,'descend');
%X1 = zeros(m,n);
for i=1:m
    X1(i,:) = X(index(i),:);
end
if(n<500)
    K=10;
    T=50;
elseif(n<1000)
    K=20;
    T=100;
elseif((n>=5000) && (n<10000) )
    K=20;
    T=100;
elseif(n>=10000)
    K=20;
    T=10;
else
    K=20;
    T=round(n/10);
end
if(m<=10000)
    XX="50, 100, 150, 200, 250";
elseif(m<=12000)
    XX="50, 100, 200, 400, 800";
else
    XX="200, 400, 600, 800, 1000";
end
disp(['*Note:  K=',num2str(K),',  T=',num2str(T),',  X1~X5 = ',num2str(XX)]);
end