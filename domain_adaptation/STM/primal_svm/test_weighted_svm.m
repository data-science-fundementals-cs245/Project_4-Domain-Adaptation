% Test standard SVM and equally-weighted SVM
% Codes for training SVM in the primal is modeified from 
%   Olivier Chapelle (olivier.chapelle@tuebingen.mpg.de)
%   http://olivier.chapelle.cc/primal/
%
% Wen-Sheng Chu (wschu@cmu.edu)
% Updated Jun-08-13

close all; clc; addpath(genpath('func'));
global X K;

% init data
n  = 5000;
d  = 500;
X0 = randn(n,d);

X  = X0; 
Y  = sign(X(:,1)); 
K  = X*X';

% init weights
gamma = ones(n,1);

% init parameters
opt.iter_max_Newton = 1e3;
lambda   = 1;
isLinear = 1;

% standard SVM with Newton
tic;
[w0,b0,obj0,sv0] = primal_svm(isLinear,Y,lambda,opt);
times(1) = toc;


% standard SVM with CG
opt.cg = 1;
tic;
[w1,b1,obj1,sv1] = primal_svm(isLinear,Y,lambda,opt);
times(2) = toc;

% equally-weighted SVM with Newton
opt.cg = 0; tic;
[model,obj2] = primal_svm_weighted(isLinear,Y,lambda,gamma);
times(3) = toc;
w2  = model.w;
b2  = model.b;
sv2 = model.sv;

% equally-weighted SVM with CG
opt.cg = 1; tic;
[model,obj3] = primal_svm_weighted(isLinear,Y,lambda,gamma,opt);
times(4) = toc;
w3  = model.w;
b3  = model.b;
sv3 = model.sv;

% print comparison
fprintf('Standard SVM with Newton in %.3f seconds\n',times(1));
fprintf('Standard SVM with CG in %.3f seconds\n',    times(2));
fprintf('Weighted SVM with Newton in %.3f seconds\n',times(3));
fprintf('Weighted SVM with CG in %.3f seconds\n',    times(4));
fprintf('obj difference using Newton=%.20f\n',obj0-obj2);
fprintf('obj difference using CG=%.6f\n',    obj1-obj3);
fprintf('#SV difference using Newton=%d\n',numel(getDiffSet(sv0,sv2)));
fprintf('#SV difference using CG=%d\n',    numel(getDiffSet(sv1,sv3)));
fprintf(' w  difference using Newton=%.6f\n',sum(abs(w0-w2))/d);
fprintf(' w  difference using CG=%.6f\n',    sum(abs(w1-w3))/d);