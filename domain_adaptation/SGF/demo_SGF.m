% clear all;clc;

addpath('.\utils');
addpath('.\tools\libsvm-3.17\matlab');

% parameter
param.C = 1;
param.dim = 200;
nPoints = 10;

% fprintf('loading data....\n');
% train_data = load('.\data\train_data');
% test_data = load('.\data\test_data');
     
XA = train_data.train_features'; 
XB = test_data.test_features';

% main algorithm
fprintf('performing SGF....\n');
sgf_model = train_sgf(XA, XB, param.dim, nPoints);
train_feature = XA * sgf_model.G;
test_feature = XB * sgf_model.G;

save('train_feature.mat', 'train_feature')
save('test_feature.mat', 'test_feature')

kparam = struct();
kparam.kernel_type = 'gaussian';
[K, kernel_param] = getKernel(train_feature',train_feature',kparam);
test_kernel = getKernel(test_feature',train_feature',kernel_param);

train_kernel    = [(1:size(K, 1))' K];
para   = sprintf('-c %.6f -s %d -t %d -w1 %.6f -q 1',param.C,0,4,1);
model  = svmtrain(train_data.train_labels, train_kernel, para);

ay      = full(model.sv_coef)*model.Label(1);
idx     = full(model.SVs);
b       = -(model.rho*model.Label(1));

decs    = test_kernel(:, idx)*ay + b;   
ap      = calc_ap(test_data.test_labels, decs);




