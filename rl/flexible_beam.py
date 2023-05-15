import numpy as np
from numpy.random.mtrand import random
from scipy import integrate
from matplotlib import pyplot as plt
from copy import deepcopy

class FlexbleBeamFixLin:
    def __init__(self) -> None:

        # rl system define
        self.min_action = -2
        self.max_action = 2
        self.max_phi = np.radians(160)
        self.min_phi = np.radians(-160)
        self.dt = 0.05
        self.total_T = 10
        self.total_iteration = int(self.total_T/self.dt)
        self.iteration = 0
        self.score_scale = 3

        # define dynamic system
        self.dynamic_sys_init()

        # target phi
        self.phi_des = np.pi/2
        # target x y
        self.x_des = self.beam_L*np.cos(self.phi_des)
        self.y_des = self.beam_L*np.sin(self.phi_des)

    def dynamic_sys_init(self):
        self.beam_L = 1.1
        self.EI = 2.1*1.167 # Young's Modulus
        self.zho = 0.4436 # density
        self.rA = 0.05; # radius of the base
        self.MA = 50 # mass of the base
        self.IH = 0.5*50*(self.rA**2); # inertia of the base
        self.c = (self.beam_L**3/3+self.IH/self.zho)**(-1/2)

        # read k and omega
        data_dir = 'linear_fixed_data/'

        self.k = np.loadtxt(data_dir+'ks.csv',delimiter=',')
        self.omegas = np.loadtxt(data_dir+'omegas.csv',delimiter=',')
        self.An = np.loadtxt(data_dir+'Ann.csv',delimiter=',')
        self.Bn = np.loadtxt(data_dir+'Bnn.csv',delimiter=',')
        self.Cn = np.loadtxt(data_dir+'Cnn.csv',delimiter=',')
        self.Dn = np.loadtxt(data_dir+'Dnn.csv',delimiter=',')

        Omega = np.diag(self.omegas)
        qD = len(self.k)
        self.qD=qD
        # A matrix
        self.sys_A = np.hstack((np.zeros((qD,qD)),np.eye(qD)))
        lowerA = np.hstack((-1*np.matmul(Omega,Omega),np.zeros((qD,qD))))
        self.sys_A = np.vstack((self.sys_A,lowerA))
        # B matrix
        self.sys_B = np.zeros((2*qD,1))
        for i in range(qD):
            if i==0:
                self.sys_B[qD+i,0]=self.Cn[i]/self.zho
            else:
                self.sys_B[qD+i,0]=self.sci_p_fcn(i,0)/self.zho
        
        # K for PD control baseline
        Kp=1
        Kd=1
        self.KPD = np.zeros((1,2*qD))
        self.KPD[0,:] = np.append(Kp*self.sys_B[qD:,0],Kd*self.sys_B[qD:,0])

    def step(self,action):

        torque = min(max(action[0],-2),2)

        # integrate over the system
        if (self.last_phi>self.max_phi and torque>0) or (self.last_phi<self.min_phi and torque<0):
            # freeze motion if phi too large
            qout = deepcopy(self.qstate)
        else:
            t = np.array([0,self.dt])
            qout = integrate.odeint(self.sys_ode,self.qstate,t,args=tuple([torque]))
            qout = qout[-1]
            self.qstate = deepcopy(qout)

        qoutdot = qout[self.qD:]
        qout = qout[:self.qD]
        qout[0] = qout[0]+self.phi_des/self.c
        l_sample = np.array([self.beam_L])
        w,phi = self.q2wphi(qout,l_sample)
        wdot,phidot = self.q2wphi(qoutdot,l_sample)
        tipx,tipy,tipxb,tipyb = self.wphi2xy(w,phi,l_sample)

        reward = -1*(np.linalg.norm([tipx-self.x_des,tipy-self.y_des])*self.score_scale+0.001 * (action[0]**2))

        # observe tip deviation and phi
        # obs_state = np.array([w[-1]*10,phi])
        obs_state = np.array([w[-1],phi,wdot[-1],phidot])

        self.iteration += 1
        done = False
        if self.iteration > self.total_iteration:
            done = True
        
        self.last_phi = phi

        return obs_state,reward,done
    
    def reset(self,random=True):

        self.qstate = np.zeros((2*self.qD))
        if random:
            dtheta_init = np.random.uniform(np.radians(-10),np.radians(60))
        else:
            dtheta_init = 0
        self.qstate[0] = (-self.phi_des+dtheta_init)/self.c
        self.iteration = 0
        self.last_phi = 0

        obs_state = np.array([0,dtheta_init])
        return obs_state

    def sys_ode(self,q_init,t,u):

        dqdt = np.matmul(self.sys_A,q_init) + self.sys_B.flatten()*u
        return dqdt

    def sys_ode_feedback(self,q_init,t):

        dqdt = np.matmul(self.sys_A-np.matmul(self.sys_B,self.KPD),q_init)
        return dqdt

    def wphi2xy(self,w,phi,l):

        x=[]
        y=[]
        xb=[]
        yb=[]
        for i in range(len(l)):
            xb.append(l[i]*np.cos(phi))
            yb.append(l[i]*np.sin(phi))
            x.append(xb[-1]-w[i]*np.sin(phi))
            y.append(yb[-1]+w[i]*np.cos(phi))
        return np.array(x),np.array(y),np.array(xb),np.array(yb)

    def q2wphi(self,q,l):

        # evaluate phi and w
        v_prime=0
        v = np.zeros(l.size)
        for i in range(self.qD):
            if i==0:
                v_prime = v_prime+q[i]*self.Cn[i]
                v = v+q[i]*self.Cn[i]*l
            else:
                v_prime = v_prime+q[i]*self.sci_p_fcn(i,0)
                v = v+q[i]*self.sci_fcn(i,l)
        phi = v_prime
        w = v-l*phi
        return w,phi

    def sci_fcn(self,ind,l):
        an = self.An[ind]
        bn = self.Bn[ind]
        cn = self.Cn[ind]
        dn = self.Dn[ind]
        k = self.k[ind]
        return an*np.sin(k*l) + bn*np.sinh(k*l) +\
             cn*np.cos(k*l) + dn*np.cosh(k*l)

    def sci_p_fcn(self,ind,l):
        an = self.An[ind]
        bn = self.Bn[ind]
        cn = self.Cn[ind]
        dn = self.Dn[ind]
        k = self.k[ind]
        return an*k*np.cos(k*l) + bn*k*np.cosh(k*l)\
             - cn*k*np.sin(k*l) + dn*k*np.sinh(k*l)

def test_PD():
    # test system
    fb = FlexbleBeamFixLin()
    # Get timesteps
    delta_t=fb.dt
    t_max=10
    t = np.linspace(0, t_max, int(t_max/delta_t))
    q0 = np.zeros((2*fb.qD))
    q0[0] = -fb.phi_des/fb.c
    qout = integrate.odeint(fb.sys_ode_feedback,q0,t)
    u = np.matmul(fb.KPD,qout.T)
    plt.plot(np.squeeze(u))
    plt.show()

def test_env():
    
    env = FlexbleBeamFixLin()
    obs_state = env.reset(random=False)
    all_w = [obs_state[0]]
    all_phi = [obs_state[1]]
    all_torques = []
    rewards = []
    while True:
        # test PD
        torque = np.squeeze(-1*np.matmul(env.KPD,env.qstate))
        torque = 2
        if len(all_torques)>100:
            torque = -2
        obs_state_next,reward,done = env.step([torque])
        print(obs_state_next)
        all_torques.append(torque)
        all_w.append(obs_state_next[0])
        all_phi.append(obs_state_next[1])
        rewards.append(reward)
        if done:
            break
    
    plt.plot(all_phi,label='phi')
    plt.plot(all_w,label='deviation w')
    plt.plot(rewards,label='instant reward')
    plt.legend()
    plt.title("PD Agent Performance")
    plt.show()

    plt.plot(all_torques,label='torque')
    plt.legend()
    plt.title("PD Agent Action")
    plt.show()

    print("PD Rewards:",np.sum(rewards))

if __name__=='__main__':

    # test_PD()
    test_env()