function out = GFK_get_metric(XA, XB, tol_eps, dim)
%

% XA : ns x d
% XB : nt x d
% min_dim : minimum dimension
% max_dim : maximum dimension

% dim = size(XA,2);

% [Ps, pre_XA] = calc_pca(XA, dim);
% [Pt, pre_XB] = calc_pca(XB, dim);
% [Pst] = calc_pca([XA; XB], dim);

[Ps, pre_XA] = calc_pca(XA');
[Pt, pre_XB] = calc_pca(XB');
[Pst] = calc_pca([XA; XB]');

% if nargin < 4
%     optimal_dim = calc_optimal_dim(Ps, Pt, Pst, tol_eps);
% else
%     optimal_dim = calc_optimal_dim(Ps, Pt, Pst, tol_eps, min_dim, max_dim);
% end

optimal_dim = dim;

Rs = null(Ps');
Q = [Ps,Rs];
Pt = Pt(:,1:min(size(Pt,2),optimal_dim));
G = GFK(Q,Pt);
% Input: Q = [Ps, null(Ps')], where Ps is the source subspace, column-wise orthonormal
%        Pt: target subsapce, column-wise orthonormal, D-by-d, d < 0.5*D
out.G = G;
out.pre_XA = pre_XA;
out.pre_XB = pre_XB;



function dim = calc_optimal_dim(Ps,Pt,Pst,tol_eps, min_dim, max_dim)
if nargin < 5
    min_dim = 1;
    max_dim = Inf;
end
max_dim = min(max_dim, min(size(Pt,2),size(Ps,2)));
min_dim = min(min_dim, max_dim);
for d = max(5,min_dim) : 5 : max_dim
    [Us,Sigmas,Vs] = svd(Ps(:,1:d)'*Pst(:,1:d));
    cos_alpha= diag(Sigmas);
    [cos_alpha, idx] = sort(cos_alpha, 'descend');
    [Ut,Sigmat,Vt] = svd(Pt(:,1:d)'*Pst(:,1:d));
    cos_beta= diag(Sigmat);
    [cos_beta, idx] = sort(cos_beta, 'descend');
    
    sin_alpha_d = real(sqrt(1-cos_alpha(d).^2));
    sin_beta_d= real(sqrt(1-cos_beta(d).^2));
    D(d) = 0.5*( sin_alpha_d + sin_beta_d);
    if D(d) >= 1-tol_eps
        break;
    end
end
dim = d;
fprintf('optimal dimension = %d\n', dim);
