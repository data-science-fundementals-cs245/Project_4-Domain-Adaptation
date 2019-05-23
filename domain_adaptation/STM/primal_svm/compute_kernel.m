function K = compute_kernel(ind1,ind2,hp)

global X
switch hp.type
  case 'linear'
    if isempty(ind2)
      K = sum(X(ind1,:).^2,2);
      return;
    end;
    K = X(ind1,:)*X(ind2,:)';

  case 'rbf'
    if isempty(ind2)
      K = ones(length(ind1),1);
      return;
    end;
    normX = sum(X(ind1,:).^2,2);
    normY = sum(X(ind2,:).^2,2);
    K = exp(-0.5/hp.sig^2*(repmat(normX ,1,length(ind2)) + ...
    repmat(normY',length(ind1),1) - ...
    2*X(ind1,:)*X(ind2,:)'));
  
  otherwise
    error('Unknown kernel');
end
