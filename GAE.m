function [allhx, allhx1, Ws] = GAE(xx, layers, A_n,lambda)

% xx : dxn input
% noise: corruption level
% layers: number of layers to stack
% A_n: d*d structure

% allhx: (layers*d)xn stacked hidden representations

% lambda = 1e-5;
disp('stacking hidden layers...')
prevhx = xx;
allhx=cell(1,layers);
allhx1 = [];
Ws={};
for layer = 1:layers
    disp(['layer:',num2str(layer)])
	tic
	[newhx, W] = GASingle(prevhx,lambda,A_n);
	Ws{layer} = W;
	toc
    allhx{layer} = newhx;
 	allhx1 = [allhx1; newhx];%存储的是中间层的嵌入特征输出
	prevhx = newhx;
end
