function  [w,obj,sv] = primal_svm_linear_weighted(Y,lambda,gamma,opt) 
% -------------------------------
% Train a linear SVM using Newton 
% -------------------------------

  global X;
  [n,d] = size(X);
    
  w = zeros(d+1,1); % The last component of w is b.
  iter = 0;
  out = ones(n,1); % Vector containing 1-Y.*(X*w)
  
  while 1
    iter = iter + 1;
    if iter > opt.iter_max_Newton;
      warning(sprintf(['Maximum number of Newton steps reached.' ...
                       'Try larger lambda']));
      break;
    end;
    
    % vincent: weigheted obj and grad
    [obj, grad, sv] = obj_fun_linear_weighted(w,Y,lambda,out,gamma);
    
    % Compute the Newton direction either exactly or by linear CG
    if ~opt.lin_cg
      % Advantage of linear CG when using sparse input: the Hessian is never computed explicitly.
      [step, foo, relres] = minres(@hess_vect_mult, -grad,...
                                   opt.cg_prec,opt.cg_it,[],[],[],sv,lambda);
    else
      Xsv = X(sv,:);
      
      %=== unweighted Hessian
      %hess0 = lambda*diag([ones(d,1); 0]) + [Xsv'*Xsv sum(Xsv,1)']; [sum(Xsv) length(sv)]];


      %*********** Notice ******************************************************
      % There is an issue in Matlab for computing the values Xsv'*Xsv and
      % Xsv'*wXsv. Even though Xsv is exactly the same as wXsv, the inner 
      % implementation in Matlab causes differences between these two values,
      % and therefore make slight different solution in the SVM solution.
      %*************************************************************************
      
      %wXsv = diag(gamma(sv))*Xsv;
      wXsv = bsxfun(@times,Xsv,gamma(sv));
      hess = lambda*diag([ones(d,1); 0]) + ...   % Hessian
             [[Xsv(:,1:end)'*wXsv sum(wXsv,1)']; [sum(wXsv) sum(gamma(sv))]];
           
      % if sum(abs(hess0(:)-hess(:)))~=0
      %   keyboard
      % end

      %step = - inv(hess)*grad;   % Newton direction
      step  = - hess\grad; % Newton direction
    end
    
    % vincent: weigheted linear search
    [t,out] = line_search_linear_weighted(w,step,out,Y,lambda,gamma);
    
    w = w + t*step;
    fprintf(['Iter = %d, Obj = %f, Nb of sv = %d, Newton decr = %.6f, ' ...
             'Line search = %.3f'],iter,obj,length(sv),-step'*grad/2,t);
%     if opt.lin_cg
%         fprintf(', Lin CG acc = %.4f     \n',relres);
%     else
%         fprintf('      \n');
%     end;
    
    if -step'*grad < opt.prec * obj  
      % Stop when the Newton decrement is small enough
      break;
    end;
  end;
  