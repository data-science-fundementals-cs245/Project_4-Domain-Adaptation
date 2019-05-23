%==========================================================================
% vincent: weighted linear objective func
function [obj, grad, sv] = obj_fun_linear_weighted(w,Y,lambda,out,gamma)
  % Compute the objective function, its gradient and the set of support vectors
  % Out is supposed to contain 1-Y.*(X*w)
  global X
  out = max(0,out); % remove negative values
  w0 = w; w0(end) = 0;  % Do not penalize b
  
  %=== compute unweighted obj and grad
  %obj = sum(out.^2)/2 + lambda*w0'*w0/2; % L2 penalization of the errors
  %grad = lambda*w0 - [((out.*Y)'*X)'; sum(out.*Y)]; % Gradient
  
  %=== compute weighted objective
  %obj = out'*diag(gamma)*out/2 + lambda*(w0'*w0)/2; % L2 penalization of the weighted errors
  %grad = lambda*w0 - [((out.*Y)'*diag(gamma)*X)'; sum(diag(gamma)*out.*Y)]; % Gradient
  
  
  %=== compute weighted object with improved speed
  spg = diag(sparse(gamma));
  obj = out'*spg*out/2 + lambda*(w0'*w0)/2; % L2 penalization of the weighted errors

  outY  = out.*Y;
  outYg = outY'*spg;
  grad  = lambda*w0 - [(outYg*X)'; sum(outYg,2)];
  
  sv = find(out>0);
  