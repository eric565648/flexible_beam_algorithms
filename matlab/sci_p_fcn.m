function y = sci_p_fcn(an,bn,cn,dn,k,l)
    y = an.*k.*cos(k*l) + bn.*k.*cosh(k*l) - cn.*k.*sin(k*l) + dn.*k.*sinh(k*l);
end

