clear all;clc;

addpath('.\utils');
addpath('.\primal_svm');
addpath('.\func');
addpath('C:\Program Files\Mosek\6\toolbox\r2009b');

% parameter
param.C = 1;
param.lambda = 1;

fprintf('loading data....\n');
train_data = load('.\data\train_data');
test_data = load('.\data\test_data');

global X Xte;
X = train_data.train_features';
Xte = test_data.test_features';

% main algorithm
model  = solveStm(train_data.train_labels,param);

decs   = Xte*model.w + model.b;   
ap     = calc_ap(test_data.test_labels, decs);


