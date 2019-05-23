function  [w, obj, sv] = primal_svm_linear_cg_weighted(Y,lambda,gamma,opt)
% -----------------------------------------------------
% Train a linear SVM using nonlinear conjugate gradient 
% -----------------------------------------------------
  global X;
  [n,d] = size(X);
    
  w = zeros(d+1,1); % The last component of w is b.
  iter = 0;
  out = ones(n,1); % Vector containing 1-Y.*(X*w)
  go = [X'*Y; sum(Y)];  % -gradient at w=0 
  
  s = go; % The first search direction is given by the gradient
  while 1
    iter = iter + 1;
    if iter > opt.cg_it * min(n,d)
      warning(sprintf(['Maximum number of CG iterations reached. ' ...
                       'Try larger lambda']));
      break;
    end;
     
    % Do an exact line search
    %[t,out] = line_search_linear(w,s,out,Y,lambda);
    [t,out] = line_search_linear_weighted(w,s,out,Y,lambda,gamma);
    w = w + t*s;
      
    % Compute the new gradient
    %[obj, gn, sv] = obj_fun_linear(w,Y,lambda,out); 
    [obj, gn, sv] = obj_fun_linear_weighted(w,Y,lambda,out,gamma);
    gn = -gn;
    fprintf('Iter = %d, Obj = %f, Norm of grad = %.3f     \n',iter,obj,norm(gn));
      
    % Stop when the relative decrease in the objective function is small 
    if t*s'*go < opt.prec*obj, break; end;
    
    % Flecher-Reeves update. Change 0 in 1 for Polack-Ribiere
    be = (gn'*gn - 0*gn'*go) / (go'*go);
    s = be*s+gn;
    go = gn;
  end;