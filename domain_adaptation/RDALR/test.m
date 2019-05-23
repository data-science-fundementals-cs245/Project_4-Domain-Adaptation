clear; clc;

lambda = 1;
m = 10;
n = 10;
E = rand(m,n);
F = rand(m,n);

prev_obj = lambda*calc_21norm(E)+1/2*norm(E-F,'fro')^2;

tmpE = sqrt(sum(E.^2,1));
          
idx1 = find(tmpE>lambda);
idx2 = find(tmpE<=lambda);
E(:,idx2) = 0;
E(:,idx1) = E(:,idx1)*sparse(1:length(idx1),1:length(idx1),1-lambda./tmpE(idx1));
opt_obj = lambda*calc_21norm(E)+1/2*norm(E-F,'fro')^2;

while true
    E = rand(m,n);
    obj = lambda*calc_21norm(E)+1/2*norm(E-F,'fro')^2;
    if obj<opt_obj
        fprintf('obj %f opt_obj %f\n', obj, opt_obj);
        break;
    end
end
