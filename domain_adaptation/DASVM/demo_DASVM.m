% clear all;clc;

addpath('.\utils');
addpath('.\tools\libsvm-weights-3.17\matlab');

% parameter

param.C = 10;
param.Cu = 1; % Cu should be less than C

param.Cu_max      = 10*param.Cu; % add at most rho patterns at each iteration
param.rho         = 10;   
param.max_iter    = 100;
param.max_unl_num = 5;


% fprintf('loading data....\n');
% train_data = load('.\data\train_data');
% test_data = load('.\data\test_data');

pos_features = train_data.train_features(:,train_data.train_labels==1);
neg_features = train_data.train_features(:,train_data.train_labels==-1);

% main algorithm
[model,kernel_param,training_features] = train_dasvm(pos_features, neg_features, test_data.test_features, param);

% prediction
test_kernel = getKernel(test_data.test_features, training_features, kernel_param);

ay      = full(model.sv_coef)*model.Label(1);
idx     = full(model.SVs);
b       = -(model.rho*model.Label(1));
decs    = test_kernel(:, idx)*ay + b;  
ap  = calc_ap(test_data.test_labels, decs);
  

