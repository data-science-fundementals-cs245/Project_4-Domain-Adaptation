function val = calc_nuclear_norm(M)
    [V,S,U] = svd(M);
    val = sum(S(:));