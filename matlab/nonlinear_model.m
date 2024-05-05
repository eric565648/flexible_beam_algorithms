clear all; close all;

load("flexible_beam_system.mat");



function dqdt = odefun(t,q_state,Ax,A1,a,B1,C1,C2,L1,D2,D3,E0,P,sc_x,scp_0,x,MA,L,zho,omegas)
    dim = length(q_state)/2-1;
    q = q_state(1:dim);
    qdot = q_state(dim+2:2*dim+1);
    Mq = MMat(q,Ax,A1,a,sc_x,scp_0,x);
    L2q = D2-0.5*scp_0*q'*D3*q;
    CLq = -P-2*D2*scp_0'+0.5*q'*P*q*scp_0*scp_0'-L^2/2*scp_0*scp_0';
    CTq = -2*P*q*scp_0' - scp_0*scp_0'*E0*q;
    Cqqd = CMat(q,dot,A1,B1,C1,C2,scp_0);

    bigM = [Mq -L1*q L2q; -q'*L1' MA/zho+L 0; L2q' 0 MA/zho+L];
    bigD = [Cqqd zeros(dim,1) zeros(dim,1); qdot'*CLq 0 0; qdot'*CTq 0 0];
    bigK = zeros(dim+2); bigK(1:dim,1:dim) = diag(omegas);
    bigMinv = inv(bigM);

    A = [zeros(dim+2) eye(dim+2); bigMinv*bidD bigMinv*bigK]
    B = [zeros(dim,3)]

    dqdt = (A-B*K)*q;
end
function Mq = MMat(q,Ax,A1,a,sc_x,scp_0,x,L)
    dim = length(q);
    Mq = a*eye(dim);
    Mq = Mq + vpaintegral(Ax*q*q'*Ax,0,L);
    Mq = Mq + q'*A1*q*scp_0*scp_0';
    Mq = Mq + 2*vpaintegral(q'*(sc_x-x*scp_0)*Ax,0,L)*q*scp_0';
    Mq = Mq - scp_0*q'*vpaintegral(Ax*q*sc_x',0,L);
end
function Cqqd = CMat(q,dot,A1,B1,C1,C2,scp_0)
    dim = length(q);
    Cqqd = B1-A1*q*qdot'*scp_0*scp_0';
    Cqqd = Cqqd + 2*scp_0*q'*A1*scp_0*qdot;
    Cqqd = Cqqd - 2*qdot'*scp_0*B1*q*scp_0';
    Cqqd = Cqqd + 2*kron(qdot',eye(dim))*C1*q*scp_0';
    Cqqd = Cqqd - 2*kron(qdot',eye(dim))*C2*q*scp_0';
end