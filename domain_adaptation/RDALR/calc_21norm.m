function val = calc_21norm(M)

    val = sum(sqrt(sum(M.^2,1)));

end

