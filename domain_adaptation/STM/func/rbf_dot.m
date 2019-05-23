%Radial basis function inner product
%Arthur Gretton

%Pattern input format : [pattern1 ; pattern2 ; ...]
%Output : p11*p21 p11*p22 ... ; p12*p21 ...
%Deg is kernel size

%Each ROW of input pattern is a separate data point

function [H]=rbf_dot(patterns1,patterns2,deg)

size1 = size(patterns1);
size2 = size(patterns2);

%new vectorised version
G = sum((patterns1.*patterns1),2);
H = sum((patterns2.*patterns2),2);

Q = repmat(G,1,size2(1));
R = repmat(H',size1(1),1);

try
  H = Q + R - 2*patterns1*patterns2';
catch ME
  keyboard
end

H = exp(-H/deg);