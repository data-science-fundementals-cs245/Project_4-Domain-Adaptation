clear all;clc;
config;

path.home = '..\..\ACT42_RGB_IDT_feature';
param.cate_num = 14;
param.dataset = 'ACT42';

addpath('\\155.69.146.69\NiuLi\Tool_Software\libsvm-3.17\matlab');
addpath('\\155.69.146.69\NiuLi\common_code_collection');

for src_view = 1:4
    for tgt_view = 1:4
        if src_view==tgt_view
            continue;
        end
        fprintf('src_view %d tgt_view %d: loading data....\n', src_view, tgt_view);
        t1 = tic;
        load(fullfile(path.home,['data_baseline_view',num2str(min(src_view,tgt_view)),num2str(max(src_view,tgt_view)),'_K8.mat']), 'pca_features', 'views', 'labels');
        toc(t1);
        src_dm_idx = find(views==src_view);
        tgt_dm_idx = find(views==tgt_view);
        train_label = labels(:,src_dm_idx);
        tr_label = sum(sparse(1:param.cate_num,1:param.cate_num,1:param.cate_num)*train_label,1)';
        te_label = labels(:,tgt_dm_idx);
        te_label = sum(sparse(1:param.cate_num,1:param.cate_num,1:param.cate_num)*te_label,1)';

        n = length(tr_label);
        p = length(te_label);

        projection_matrix = calc_pca(pca_features);
        pca_features = projection_matrix'*pca_features;
        d = size(pca_features,1);
        S = pca_features(:,src_dm_idx);
        T = pca_features(:,tgt_dm_idx);
        
        for ai = 1:length(alphas)
            for bi = 1:length(betas)
                param.alpha = alphas(ai);
                param.beta = betas(bi);
                
                save_file_name = ['.\output\',param.dataset,'_src_',num2str(src_view),'_tgt_',num2str(tgt_view),...
                    '_alpha_',num2str(param.alpha),'_beta_',num2str(param.beta),'.mat'];
                if exist(save_file_name, 'file')
                    continue;
                else
                    save(save_file_name, 'param', '-v7.3');
                end
        
                Q = zeros(d,n);
                J = zeros(d,n);
                E = zeros(d,n);
                Y = zeros(p,n);
                U = zeros(d,n);
                V = zeros(d,n);
                Z = zeros(p,n);
                F = zeros(p,n);
                W = eye(d);
                mu = 1e-7;

                epsilon = 1e-3;
                max_iter = 10;
                
                for iter = 1:max_iter
                    fprintf('%s iter %d.....\n', save_file_name, iter);
                  
                    % Update Z
                    Z = (eye(p)+T'*T)\(T'*(W*S-E)+1/mu*(T'*V-Y)+F);

                    % Update J
                    [left,sigma,right] = svd(W*S+U/mu);
                    sigma = sigma-param.beta/mu;           
                    sigma(sigma<0) = 0;
                    J = left*sigma*right';

                    % Update E
                    tmpE = sqrt(sum(E.^2,1));
                    lambda = param.alpha/mu;
                    idx1 = find(tmpE>lambda);
                    idx2 = find(tmpE<=lambda);
                    E(:,idx2) = 0;
                    E(:,idx1) = E(:,idx1)*sparse(1:length(idx1),1:length(idx1),1-lambda./tmpE(idx1));
                    
                     % Update F    
                    [left,sigma,right] = svd(Z+Y/mu);
                    sigma = sigma-param.beta/mu;           
                    sigma(sigma<0) = 0;
                    F = left*sigma*right';
                    
                    % Update W
                    tmpS = S*S';
                    W = ((J+T*Z+E)*S'-1/mu*(U+V)*S')/(tmpS+1e-10*mean(diag(tmpS))*eye(d));
                    W = orth(W);
                    W(:,size(W,2)+1:d) = 0;
                    

                    % Update Multipliers
                    Y = Y+mu*(Z-F);
                    U = U+mu*(W*S-J);
                    V = V+mu*(W*S-T*Z-E);

                    % update mu
                    mu = min(1.2*mu, 10^10);

                    err1 = norm(Z-F);
                    err2 = norm(W*S-J);
                    err3 = norm(W*S-T*Z-E);
                    fprintf('err %f %f %f\n', err1, err2, err3);
                    if err1<epsilon && err2<epsilon && err3<epsilon
                        break;
                    end
                end

                train_feature = W*S-E;
                test_feature = T;

                acc_arr = zeros(length(lambdas),1);
                for li = 1:length(lambdas)
                    param.lambda = lambdas(li);

                    W = (train_label*train_feature')/(train_feature*train_feature'+param.lambda*eye(size(train_feature,1)));
                    test_decs = test_feature'*W';
                    [~,y_pred] = max(test_decs,[],2);
                    [~,~,acc] = calc_confusion_matrix(y_pred, te_label);
                    acc_arr(li) = acc;
                end
                save(save_file_name, 'acc_arr', '-v7.3');
     
            end
        end
    end
end








