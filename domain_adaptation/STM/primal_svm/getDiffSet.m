function diffset = getDiffSet(set1,set2)
% get different elements between set1 and set2

d1 = setdiff(set1,set2);
d2 = setdiff(set2,set1);
diffset = [d1(:); d2(:)];