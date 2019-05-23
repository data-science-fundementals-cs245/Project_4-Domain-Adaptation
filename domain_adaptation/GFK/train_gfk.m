function G = train_gfk(XA, XB, tol_eps, dim)

% XA :  nA x d
% XB : nB x d
% we assume the data has already been centered

tt = tic;
fprintf('GFK_get_metric ...');
common_GFK = GFK_get_metric(XA, XB, tol_eps, dim);
fprintf('finished using %g sec.\n', toc(tt));

G = common_GFK.G;

end
