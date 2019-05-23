function [W, ee] = train_sstca(K_c_ss, K_c_su, K_c_uu, X_c_s, X_c_u, ys, gamma, lambda, mu)

ns = size(K_c_ss,1);
nu = size(K_c_su,2);
n  = ns + nu;

K = [
    K_c_ss  K_c_su
    K_c_su' K_c_uu
    ];


if lambda>0
    options.NN = 5;
    options.GraphDistanceFunction = 'euclidean';
    options.GraphWeights = 'heat';
    options.GraphWeightParam = 0;
    options.LaplacianNormalize = 1;
    options.LaplacianDegree = 1;
    tt = tic;
    L = laplacian(options, [X_c_s; X_c_u]);
    fprintf('Computing Laplacian matrix using %g sec.\n', toc(tt));
else
    L = zeros(n);
end

s = [ones(ns,1)/ns; -ones(nu,1)/nu];

Kyy = zeros(ns+nu);
Kyy(1:ns,1:ns) = 2 * double(repmat(ys,1,ns) == repmat(ys', ns, 1)) - 1;
Kyy = gamma * Kyy + (1-gamma)*eye(ns+nu);

H = eye(ns+nu) - (1/(ns+nu));
tt = tic;
B = K*(s*s' + lambda*L)*K + mu * eye(ns+nu);
D = (K*H*Kyy*H*K);
M = B\D; % inverted B multiplies D

fprintf('Computing M using %g sec.\n', toc(tt));

tt = tic;
[tmp, ee] = eig(M);
fprintf('eig using %g sec.\n', toc(tt));

tmp = real(tmp); ee = real(diag(ee));
[ee, idx] = sort(ee, 'descend');
W = tmp(:,idx);