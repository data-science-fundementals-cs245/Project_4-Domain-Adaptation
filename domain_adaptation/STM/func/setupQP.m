function [K,kappa,A,b,Aeq,beq,LB,UB,opts] = setupQP

global X Xte;

% set kernel size
dist  = pdist2(X,X);
sigma = sqrt(median(dist(:))); 

% compute kernel
K = rbf_dot(X,X,sigma);
K = (K+K')/2; %make the matrix symmetric (it isn't symmetric before because of bad precision)

% init
nTr = size(X,1);
nTe = size(Xte,1);

% compute kappa
dist  = pdist2(X,Xte);
sigma = sqrt(median(dist(:)))
R3    = rbf_dot(X,Xte,sigma);
kappa = (R3*ones(nTe, 1));
kappa = nTr/nTe*kappa;

% compute QP variables
eps = (sqrt(nTr)-1)/sqrt(nTr);
% eps=1000/sqrt(nTr);
A=ones(1,nTr);
A(2,:)=-ones(1,nTr);
b=[nTr*(eps+1); nTr*(eps-1)];

Aeq = [];
beq = [];

% 0 <= beta_i <= 2000 for all i
LB = zeros(nTr,1);
UB = ones(nTr,1).*2000;

% opti settings
opts.MaxIter   = 3000;
opts.Algorithm = 'active-set';