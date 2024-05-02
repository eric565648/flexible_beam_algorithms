clear all; close all;

load("flexible_beam_system.mat");

function y = sci_fcn(an,bn,cn,dn,k,l)
    y = an.*sin(k*l) + bn.*sinh(k*l) + cn.*cos(k*l) + dn.*cosh(k*l);
end
function y = sci_p_fcn(an,bn,cn,dn,k,l)
    y = an.*k.*cos(k*l) + bn.*k.*cosh(k*l) - cn.*k.*sin(k*l) + dn.*k.*sinh(k*l);
end