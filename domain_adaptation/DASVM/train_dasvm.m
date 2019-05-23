function [svm_model, kernel_param, training_features] = train_dasvm(pos_features, neg_features, ul_features, param)
% function [model models]= DASVMtrain(pos_features, neg_features, ul_features, param)
% Input:
%   pos_features    - source domain positive samples, d-by-p 
%   neg_features    - source domain negative samples, d-by-n 
%   ul_features     - target domain samples, d-by-u 
%   param           - params
%
% Output:
%   model           - the final model models  - the mode list


s_features  = [pos_features neg_features];

% initialization
C           = param.C;
Cu          = param.Cu;
Cu_max      = param.Cu_max;
% add at most rho patterns at each iteration
rho         = param.rho;   
MAX_ITER    = param.max_iter;
umax        = param.max_unl_num;

pos_idx     = 1:size(pos_features, 2);
neg_idx     = size(pos_features, 2)+1:size(s_features, 2);
ul_pos_idx  = [];
ul_neg_idx  = [];

% models      = cell(0);
s_pool  = 1:size(s_features, 2);

for i = 1:MAX_ITER
    % collect positive and negative training samples
    s_pos_feats = s_features(:, pos_idx);
    s_neg_feats = s_features(:, neg_idx);
    t_pos_feats = ul_features(:, ul_pos_idx);
    t_neg_feats = ul_features(:, ul_neg_idx);
    
    % compute new svm_C and svm_Cu
    sp.svm_C    = max(C+(Cu-C)*(i-1)^2/MAX_ITER^2, Cu);
    sp.svm_Cu   = Cu+(Cu_max - Cu)*(i-1)^2/(MAX_ITER-1)^2;
    fprintf('Iter #%3d: s_pos = %d s_neg = %d, t_pos = %d t_neg = %d.\n', i, length(pos_idx), length(neg_idx), length(ul_pos_idx), length(ul_neg_idx));
    fprintf('\t\t C = %f,\t C* = %f.\n', sp.svm_C, sp.svm_Cu);
    
    pos_sn  = size(s_pos_feats, 2);
    neg_sn  = size(s_neg_feats, 2);
    pos_tn  = size(t_pos_feats, 2);
    neg_tn  = size(t_neg_feats, 2);

    training_features   = [s_pos_feats s_neg_feats t_pos_feats t_neg_feats];
    training_labels     = [ones(pos_sn, 1); -ones(neg_sn, 1); ones(pos_tn, 1); -ones(neg_tn, 1)];

    % train SVM
    vparam.kernel_type = 'gaussian';
    [training_kernel,kernel_param] = getKernel(training_features, vparam);
    training_weights = [sp.svm_C * ones(pos_sn+neg_sn, 1); sp.svm_Cu * ones(pos_tn+neg_tn, 1)];
    svm_model = svmtrain(training_weights, training_labels, [(1:length(training_labels))', training_kernel], '-t 4 -c 1 -q');
    
    % check converge criterion
    if(i ~=1 && (isempty(cand)|| H_size < umax ))
        fprintf('Stop criteria reached, break.\n');
        break;
    end

    if(i == MAX_ITER)
        fprintf('Max ites reached, break!\n');
        break;
    end
    % use generated model to test labeled source samples and target
    % samples, s_features (source samples) and ul_features (target samples)
    % are fixed
    s_test_kernel = getKernel(s_features(:, s_pool), training_features, kernel_param);
    u_test_kernel = getKernel(ul_features, training_features, kernel_param);
    ay      = full(svm_model.sv_coef)*svm_model.Label(1);
    idx     = full(svm_model.SVs);
    b       = -(svm_model.rho*svm_model.Label(1));
    sdecs    = s_test_kernel(:, idx)*ay + b;   
    udecs    = u_test_kernel(:, idx)*ay + b;  
    

    % 1. find addable unlabeled target samples
    ul_pool     = setdiff(1:size(ul_features, 2), [ul_pos_idx ul_neg_idx]);
    [tmp, sidx]  = sort(udecs(ul_pool), 'descend');
    
    % add at most rho unlabeled target samples in each iteration H set
    pos_cand    = sidx(tmp<1&tmp>0);    
    pos_cn      = min(length(pos_cand), rho);
    pos_cand    = pos_cand(1:pos_cn);    
    
    neg_cand    = sidx(tmp>-1& tmp<=0);
    neg_cn      = min(length(neg_cand), rho);
    neg_cand    = neg_cand(end+1-neg_cn:end);
    fprintf('select %d pos_cand %d neg_cand...\n', length(pos_cand), length(neg_cand));
    
    if isempty(pos_cand) || isempty(neg_cand)
        break;
    end
    
    % 2. remove some labeled target samples

    % check whether the labels of labeled target samples are the same as
    % last iteration, kick out those changed samples
    ul_pos_idx = ul_pos_idx(udecs(ul_pos_idx)>0);
    ul_neg_idx = ul_neg_idx(udecs(ul_neg_idx)<0);

    % add new unlabeled target samples
    H_size  = length(pos_cand) + length(neg_cand);  
    ul_pos_idx = [ul_pos_idx ul_pool(pos_cand)];    %#ok<AGROW>
    ul_neg_idx = [ul_neg_idx ul_pool(neg_cand)];    %#ok<AGROW>
    fprintf('unlabeled num %d pos %d neg...\n', length(ul_pos_idx), length(ul_neg_idx));
    
    % 3. remove some labeled source samples
    
    % calculate low and high boundary for removed source samples
    isHempty    = isempty(pos_cand)&isempty(neg_cand);
    up_dn       = length(pos_cand)*(1-isHempty) + rho*isHempty;
    low_dn      = length(neg_cand)*(1-isHempty) + rho*isHempty;    
    [tmp, sidx] = sort(sdecs, 'descend');
    
    up_del      = sidx(tmp>0);
    up_dn       = min(up_dn, length(up_del));
    low_del     = sidx(tmp<0);
    low_dn      = min(low_dn, length(low_del));
    
    cand        = [up_del(up_dn+1:end); low_del(1:end-low_dn)];
    % sperate pos and neg from cand
    pos_n       = length(pos_idx);
    pos_idx     = s_pool(cand(cand<=pos_n));

    neg_idx     = s_pool(cand(cand>pos_n));
    s_pool      = [pos_idx neg_idx];    % refine the s_pool    
    fprintf('remove %d pos %d neg, and %d+ %d- left ...\n', up_dn, low_dn, length(pos_idx), length(neg_idx));
    
    % 4. finally, if s_pool is empty
    if(isempty(s_pool))
        % check if ul_pos_idx/ul_neg_idx is empty
        if(isempty(ul_pos_idx))
            % select highest udecs as positive
            [~, idx] = max(udecs);
            ul_pos_idx = idx;
            ul_neg_idx = setdiff(ul_neg_idx, idx);
        end
        if(isempty(ul_neg_idx))
            % select lowest udecs as positive
            [~, idx] = min(udecs);
            ul_neg_idx = idx;
            ul_pos_idx = setdiff(ul_pos_idx, idx);
        end
    end
end
