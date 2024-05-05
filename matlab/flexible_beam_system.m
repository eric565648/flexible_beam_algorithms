clear all;

%%% system setup
L = 1.1; % 1.1 m
EI_ratio=10;
EI = 2.1*1.167*EI_ratio;
zho = 0.4436;
rA = 0.05; % radius of the base
MA = 50;
IH = 0.5*50*(rA^2); % inertia of the base
a=1;
% c = 1.44;
c = sqrt(a)*(L^3/3+IH/zho)^(-1/2);

% desired angle phi
phi_des = pi/4;

% N significant modes

% eigen functions
omegas = [];
ks = [];
inits = [];
total_N_limit = 8;
for init=0:0.5:10
    syms k real;
    eqn = IH/zho*k^3*(1+cos(k*L)*cosh(k*L))+(sin(k*L)*cosh(k*L)-cos(k*L)*sinh(k*L)) == 0;
    k = vpasolve(eqn,init*pi);
    omega = sqrt(EI/zho*k^4);
    
    if k<0
        continue
    elseif any(ks==double(round(k,8)))
        continue
    end
    ks = [ks double(round(k,8))];
    omegas = [omegas double(round(omega,8))];
    inits = [inits init];
    if length(omegas)>=total_N_limit
        break
    end
end
qD = length(ks);

syms l cn real;
sci_0 = c*l;
Ann=[0];
Bnn=[0];
Cnn=[c];
Dnn=[0];
for i=2:length(ks)
    k=ks(i);
    dn=-cn;
    an=(cos(k*L)*cosh(k*L)+sin(k*L)*sinh(k*L)+1)/(-sin(k*L)*cosh(k*L)+cos(k*L)*sinh(k*L))*cn;
    bn=2*zho/(IH*k^3)*cn-an;
    sci = cn*cos(k*l) + dn*cosh(k*l) + an*sin(k*l) + bn*sinh(k*l);
    sci_p_0 = an*k + bn*k;
    
    fcn=int((sci)^2,l,0,L)+IH/zho*sci_p_0*sci_p_0 == a;
    cnn = vpasolve(fcn,cn,1);
    cnn = double(cnn(2));
    dnn = -cnn;
    ann = (cos(k*L)*cosh(k*L)+sin(k*L)*sinh(k*L)+1)/(-sin(k*L)*cosh(k*L)+cos(k*L)*sinh(k*L))*cnn;
    bnn = 2*zho/(IH*k^3)*cnn-ann;
    
    Cnn = [Cnn cnn];
    Dnn = [Dnn dnn];
    Ann = [Ann ann];
    Bnn = [Bnn bnn];
end
%%%%%%

% pre-calculate all integration
syms xi x real;
scp_0 = sci_p_fcn(Ann,Bnn,Cnn,Dnn,ks,0)';
scp_xi = sci_p_fcn(Ann,Bnn,Cnn,Dnn,ks,xi)';
scp_x = sci_p_fcn(Ann,Bnn,Cnn,Dnn,ks,x)';
sc_0 = sci_p_fcn(Ann,Bnn,Cnn,Dnn,ks,0)';
sc_xi = sci_p_fcn(Ann,Bnn,Cnn,Dnn,ks,xi)';
sc_x = sci_p_fcn(Ann,Bnn,Cnn,Dnn,ks,x)';
scpxi_m_scp0_outer = (scp_xi-scp_0)*(scp_xi-scp_0)';
scxi_m_xiscp0_outer = (sc_xi-xi*scp_0)*(sc_xi-xi*scp_0)';
Ax = vpaintegral(scpxi_m_scp0_outer,0,x); disp('Ax');
F0 = vpaintegral((sc_x-x*scp_0)*Ax,0,L);
A1 = vpaintegral(scxi_m_xiscp0_outer,0,L); disp('A1');
B1 = vpaintegral((L^2/2-xi^2/2)*scpxi_m_scp0_outer,0,L); disp('B1');
C0 = vpaintegral(sc_xi,x,L); disp('C0');
C1 = vpaintegral(kron(C0,(scp_x-scp_0))*(scp_x-scp_0)',0,L); disp('C1');
C2 = vpaintegral(kron((scp_x-scp_0),C0)*(scp_x-scp_0)',0,L); disp('C2');
L1 = vpaintegral(Ax+scp_0*(sc_x'-x*scp_0'),0,L); disp('L1');
D2 = vpaintegral(sc_x,0,L); D3 = vpaintegral(Ax,0,L); disp('D2 D3');
E0 = vpaintegral(sc_x-x*scp_0,0,L); disp('E0');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

save("flexible_beam_system.mat");