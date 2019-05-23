function [model,gamma,log] = solveStm(Y, param)

global X;
if ~exist('param','var'), param = []; end
lambda = ps(param,'lambda',10);
C      = ps(param,'C',10);

% init 
nTr = size(X,1);
xi   = zeros(nTr,1);
iter = 0;
not_converge = 1;

% Objective
obj = inline('.5*lambdaWSVM*w''*w + .5*xi''*gamma + lambda * (.5*gamma''*K*gamma - kappa''*gamma)',...
             'w','lambdaWSVM','gamma','xi','lambda','K','kappa');

% Setup QP subproblem
[K,kappa,A,b,Aeq,beq,LB,UB,opts] = setupQP;

% Make K more numerically stable
K = K + eye(nTr)*1e-10; 

while not_converge
  iter = iter + 1;
  
  % Solve gamma
  f     = xi/2/lambda - kappa;
  gamma = quadprog(K,f,A,b,Aeq,beq,LB,UB,[]);
  
  % Solve w with L2 penalization of the weighted errors
  % Note that in primal SVM, lambda=1/C (given C is the tradeoff in standard SVM)
  model = primal_svm_weighted(1,Y,1/C,gamma);
  
  % Update xi
  xi = max(0,1-Y.*(X*model.w + model.b)).^2;
 
  % Pack log info
  log.obj(iter)   = obj(model.w,1/C,gamma,xi,lambda/nTr/nTr,K,kappa);
  log.gamma{iter} = gamma;
  
  % Check convergence
  if iter == 1
    continue
  end
  if iter>1000
      break;
  end
  if ( log.obj(end-1)-log.obj(end)) / abs(log.obj(1) ) < 1e-4
    not_converge = 0;
  end
end
log.NumberOfIter = iter;