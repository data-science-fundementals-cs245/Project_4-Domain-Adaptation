function [t,out] = line_search_linear_weighted(w,d,out,Y,lambda,gamma) 
  % From the current solution w, do a line search in the direction d by
  % 1D Newton minimization
  global X
  t = 0;
  % Precompute some dots products
  Xd = X*d(1:end-1)+d(end);
  wd = lambda * w(1:end-1)'*d(1:end-1);
  dd = lambda * d(1:end-1)'*d(1:end-1);
  iter = 1;
  while 1
    iter = iter+1;
    out2 = out - t*(Y.*Xd); % The new outputs after a step of length t
    sv = find(out2>0);
    
    spgsv = sparse(gamma);
    
    g     = wd + t*dd - (out2.*Y.*gamma)' * Xd; % The gradient (along the line)
    h     = dd + Xd'*diag(spgsv)*Xd; % The second derivative (along the line)
    t     = t - g/h; % Take the 1D Newton step. Note that if d was an exact Newton
                     % direction, t is 1 after the first iteration.           
    if g^2/h < 1e-10, break; end;
%     if iter>10000
%         break;
%     end
  end;
  out = out2;
  