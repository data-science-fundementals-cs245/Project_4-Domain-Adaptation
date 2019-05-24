% clear all;clc;

addpath('.\utils');
addpath('.\graph');
addpath('.\tools\libsvm-3.17\matlab');

% parameter
param.C     = 1;
param.gamma = 0; % without using weighted graph laplacian
param.mu    = 1;
param.lambda = 0; % without using training labels
energy_ratio = 0.999;


% fprintf('loading data....\n');
% train_data = load('.\data\train_data');
% test_data = load('.\data\test_data');
                
X_c_s = train_data.train_features'; 
% X_c_u can be sampled test_features
X_c_u = test_data.test_features';

kparam = struct();
kparam.kernel_type = 'gaussian';
[K_c_ss,kernel_param] = getKernel(X_c_s', kparam);
K_c_su = getKernel(X_c_s', X_c_u', kernel_param);
K_c_uu = getKernel(X_c_u', kernel_param);
K_c_tu = getKernel(test_data.test_features, X_c_u', kernel_param);
K_c_ts = getKernel(test_data.test_features, X_c_s', kernel_param);
tmp_K = [K_c_ss, K_c_su; K_c_ts, K_c_tu];

[nt,ns] = size(K_c_ts);

% main algorithm
fprintf('performing TCA....\n');
[W,eig_val] = train_sstca(K_c_ss, K_c_su, K_c_uu, X_c_s, X_c_u, train_data.train_labels, param.gamma, param.lambda, param.mu);

ratio = cumsum(eig_val) / sum(eig_val);
ind = find(ratio > energy_ratio);
fprintf('dimension %d saves %f energy....\n', ind(1),ratio(ind(1)));
W  = W(:,1:ind(1));

K_tilde = tmp_K*(W*W')*tmp_K';
Kernel = K_tilde(1:ns,1:ns);
test_kernel = K_tilde(ns+1:ns+nt,1:ns);

train_kernel    = [(1:size(Kernel, 1))' Kernel];

para=sprintf('-c %.6f -s %d -t %d -w1 %.6f -q 1',param.C,0,4,1);
model  = svmtrain(train_data.train_labels, train_kernel, para);

ay      = full(model.sv_coef)*model.Label(1);
idx     = full(model.SVs);
b       = -(model.rho*model.Label(1));

decs    = test_kernel(:, idx)*ay + b;  
ap      = calc_ap(test_data.test_labels, decs);
