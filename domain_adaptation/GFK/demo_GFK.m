clear all;clc;

addpath('.\utils');
addpath('.\tools\libsvm-3.17\matlab');

% parameter
param.dim = 200;
param.C = 1;
tol_eps = 0.000001;

fprintf('loading data....\n');
train_data = load('.\data\train_data');
test_data = load('.\data\test_data');


XA = train_data.train_features';
XA = XA - repmat(mean(XA,1),size(XA,1),1);

XB = test_data.test_features';
XB = XB - repmat(mean(XA,1),size(XB,1),1);
   
fprintf('performing GFK....\n');
G = train_gfk(XA, XB, tol_eps, param.dim);
K = XA * G * XA'; 
test_kernel = XB * G * XA';

train_kernel    = [(1:size(K, 1))' K];

para   = sprintf('-c %.6f -s %d -t %d -w1 %.6f -q 1',param.C,0,4,1);
model  = svmtrain(train_data.train_labels, train_kernel, para);

ay      = full(model.sv_coef)*model.Label(1);
idx     = full(model.SVs);
b       = -(model.rho*model.Label(1));

decs    = test_kernel(:, idx)*ay + b;      

ap  = calc_ap(test_data.test_labels, decs);





