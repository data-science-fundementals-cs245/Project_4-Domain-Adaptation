function model =  train_sgf(Xs, Xt, d, nPoints)
% Xs : each row is a sample vector in the source domain
% Xt : each row is a smaple vector in the target domain
% d : number of dimensions
%nPoints: number of interpolated spaces


% model.


interop_points = linspace(0,1, nPoints);

ns = size(Xs,1);
nt = size(Xt,1);

d = min(d, nt);
d = min(d, ns);
% d = min(d, floor(size(Xs,2)/2));


Ps = calc_pca(Xs');
Ps = Ps(:,1:min(size(Ps,2),d));
Pt = calc_pca(Xt');
Pt = Pt(:,1:min(size(Ps,2),d));

fprintf('compute_velocity_grassmann_efficient ... ');
tt = tic;
A = compute_velocity_grassmann_efficient(Ps, Pt);
fprintf(' using %g sec.\n', toc(tt));

fprintf('compute_Y_havingVelocity...\n');
tt = tic;
for t = 1 : length(interop_points)
    fprintf('\t%d\n', t);
    G{t} = compute_Y_havingVelocity(Ps, A, interop_points(t));
end
fprintf(' compute_Y_havingVelocity using %g sec.\n', toc(tt));

G = cell2mat(G);


model.Ps = Ps;
model.Pt = Pt;
model.G = G;
model.M = G*G';
