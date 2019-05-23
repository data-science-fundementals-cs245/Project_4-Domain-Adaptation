function [model,obj] = primal_svm_weighted(linear,Y,lambda,gamma,opt)
% [SOL, B] = PRIMAL_SVM(LINEAR,Y,LAMBDA,OPT)
% Solves the SVM optimization problem in the primal (with quatratic
%   penalization of the training errors).  
%
% If LINEAR is 1, a global variable X containing the training inputs
%   should be defined. X is an n x d matrix (n = number of points).
% If LINEAR is 0, a global variable K (the n x n kernel matrix) should be defined.  
% Y is the target vector (+1 or -1, length n). 
% LAMBDA is the regularization parameter ( = 1/C)
%
% IF LINEAR is 0, SOL is the expansion of the solution (vector beta of length n).
% IF LINEAR is 1, SOL is the hyperplane w (vector of length d).
% B is the bias
% The outputs on the training points are either K*SOL+B or X*SOL+B
% OBJ is the objective function value
% 
% OPT is a structure containing the options (in brackets default values):
%   cg: Do not use Newton, but nonlinear conjugate gradients [0]
%   lin_cg: Compute the Newton step with linear CG 
%           [0 unless solving sparse linear SVM]
%   iter_max_Newton: Maximum number of Newton steps [20]
%   prec: Stopping criterion
%   cg_prec and cg_it: stopping criteria for the linear CG.
 
% Copyright Olivier Chapelle, olivier.chapelle@tuebingen.mpg.de
% Last modified 25/08/2006  
  
  if nargin < 5       % Assign the options to their default values
    opt = [];
  end;
  opt.cg = 0;
  if ~isfield(opt,'cg'),                opt.cg = 0;                        end;
  if ~isfield(opt,'lin_cg'),            opt.lin_cg = 0;                    end;  
  if ~isfield(opt,'iter_max_Newton'),   opt.iter_max_Newton = 20;          end;  
  if ~isfield(opt,'prec'),              opt.prec = 1e-6;                   end;  
  if ~isfield(opt,'cg_prec'),           opt.cg_prec = 1e-4;                end;  
  if ~isfield(opt,'cg_it'),             opt.cg_it = 20;                    end;  
  
  
  % Call the right function depending on problem type and CG / Newton 
  % Also check that X / K exists and that the dimension of Y is correct
  if  linear 
    global X;
    if isempty(X), error('Global variable X undefined'); end;
    [n,d] = size(X);
    if issparse(X), opt.lin_cg = 1; end;
    if size(Y,1)~=n, error('Dimension error'); end;
    if opt.cg
      [sol,obj,sv] = primal_svm_linear_weighted(Y,lambda,gamma,opt);
    else
      [sol,obj,sv] = primal_svm_linear_cg_weighted(Y,lambda,gamma,opt);
    end;
    
  else
    error('Nonlinear version will be released soon :)');
    %     global K;
    %     if isempty(K), error('Global variable K undefined'); end;
    %     n = size(Y,1);
    %     if any(size(K)~=n), error('Dimension error'); end;
    %     if ~opt.cg 
    %       [sol,obj] = primal_svm_nonlinear   (Y,lambda,opt); 
    %     else
    %       [sol,obj] = primal_svm_nonlinear_cg(Y,lambda,opt); 
    %     end;
  end;
  % The last component of the solution is the bias b.
  model.w  = sol(1:end-1);
  model.b  = sol(end);
  model.sv = sv;