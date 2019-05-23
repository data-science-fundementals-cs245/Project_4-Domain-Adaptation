function [X,Y,Xte,Yte] = genData

randn('seed',0);

nTr = 200; 
nTe = 80;
d   = 2;

% generate training samples
X   = randn(nTr,d);        
Y   = sign(X(:,1));

% generate test samples
Xte = randn(nTe,d)/2; 
Yte = double(3*Xte(:,1)+Xte(:,2)>1); Yte(~Yte) = -1;
Xte(Yte==1,:) = Xte(Yte==1,:) + .1;
Xte(Yte==1,2) = Xte(Yte==1,2) + .5;
Xte(Yte~=1,:) = Xte(Yte~=1,:) - .1;
Xte(Yte~=1,2) = Xte(Yte~=1,2) - .5;