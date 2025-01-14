function [hx, W] = GASingle(xx,lambda,A_n)
% 单层图神经网络（GCN） 输入数据xx（第一层为原始数据X），输出mapping W和潜在表示hx(xx经这一层编码后的输出)
%n:sample;d:feature
% xx : dxn input    nxd
% noise: corruption level
% lambda: regularization
% A_n : dxd structure(相当于U~*X')

% hx: dxn hidden representation
% W: dx(d+1) mapping

[d, n] = size(xx);
% adding bias   
xxb = [xx; ones(1, n)];   %n x d+1

% scatter matrix S
S = xxb*xxb';
Sp = xxb * A_n' *xxb';
Sq = xxb * A_n * A_n' * xxb';   %n x n

% corruption vector
q = ones(d+1, 1);
q(end) = 1;

% Q: (d+1)x(d+1)
Q = Sq.*(q*q');
Q(1:d+2:end) = q.*diag(Sq);

% P: dx(d+1)
P = Sp(1:end-1,:).*repmat(q', d, 1);

% final W = P*Q^-1, dx(d+1);
reg = lambda*eye(d+1);
reg(end,end) = 0;
W = P * pinv(Q+reg);% the hidden representation in the autoencoder;
%W = P/(Q+reg);

hx = W*xxb*A_n;%the reconstructed representation of the input xx;

