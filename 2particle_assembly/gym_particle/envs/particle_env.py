import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math
import numpy.matlib
from numpy import linalg as LA
import main as mf

num_particles=2
num_inlets = 6
K_RWD = 4;
RWD_TAR = 1000 # reward to gain when the point approaches the target.
RWD_OUT = -500 # rewad when the particle get out of the domain.
DIST_EPSI_GOAL = 0.004
COEF_EXCEED=-0.1
DIST_EPSI_INLET=0.12 # need to prevent the particle from too close to the inlets.
MAX_STEP=20000
DT = 0.02

ACTION_LOW = -1.0
ACTION_HIGH = 1.0
ACTION_MEAN = (ACTION_LOW + ACTION_HIGH)/2


X_ini = 0.6
X_ini2 =0.6
Y_ini = 0.2
Y_ini2 =0.2

X_tar = [-0.8 ]
Y_tar = [0.2]
X_tar2 =[-0.8 ]
Y_tar2 = [-0.8 ]



'''
def reward_exceed(q, qmax, qmin):
    reward = 0
    for i in range(q.shape[0]):
        if q[i]>qmax:
            reward += (q[i]-qmax)**2
        
        if q[i]<qmin:
            reward += (q[i]-qmin)**2

    return reward
'''
class ParticleEnv(gym.Env):
    metadata = {
        'render.modes' : ['human'],
        'video.frames_per_second' : 30
    }

    def __init__(self, num_inlets=num_inlets):
        self.ntargets = len(X_tar)
        self.max_rate = 1
        self.xmax = 1
        self.nparticles=num_particles
        self.n = num_inlets
        self.dt=DT
        self.dtt=DT
        self.viewer = None
        nobs = 2 + self.ntargets
        self.action_space = spaces.Box(low=-self.max_rate, high=self.max_rate, shape=(self.n-1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.xmax, high=self.xmax, shape=(nobs,), dtype=np.float32)
        self.X_ini = X_ini
        self.Y_ini = Y_ini
        self.X_tar = X_tar
        self.Y_tar = Y_tar
        self.X_ini2 = X_ini2
        self.Y_ini2 = Y_ini2
        self.X_tar2 = X_tar2
        self.Y_tar2 = Y_tar2
        self.dist_epsi_goal = DIST_EPSI_GOAL
        self.ACTION_LOW = ACTION_LOW
        self.ACTION_HIGH = ACTION_HIGH
        ACTION_MEAN = (self.ACTION_LOW + self.ACTION_HIGH)/2.
        self.rinlets = np.zeros((self.n,2),dtype=np.float32)
        self.seed()
        self.nstep = 1
#        self.x_tar = np.array([self.X_tar, self.Y_tar,self.X_tar2, self.Y_tar2], dtype=np.float32) ## 2 rows * ntargets columns
        self.x_tar = np.array([self.X_tar, self.Y_tar], dtype=np.float32)
        #print('herer')
        
        #print(self.x_tar)

        self.reach_targets = np.zeros(self.ntargets, dtype=int)
        self.it = 0
        phi = 2*math.pi/self.n
        for i in range(self.n):
            angle = i*phi
            self.rinlets[i,:] = np.array([math.cos(angle), math.sin(angle)],dtype=np.float32)

    def min_dist_2_inlets(self, state, rinlets):
        state_ = np.matlib.repmat(state,self.n,1)
        r      = state_ - rinlets
        d      = np.squeeze(np.linalg.norm(r,axis=1))
        ind    = np.argmin(d)
        return ind, d[ind]  ## index of the inlets cloest to the current position. 
    '''
    def cal_reward(self, state, state_next, target):
        vec1 = np.squeeze(state_next - state)/self.dt
        vec2 = np.squeeze(target - state)
        vec2 = vec2/np.linalg.norm(vec2)
        ## THIS TENDS TO MAXIMIZE THE VELOCITY AND HENCE THE PARTICLE WOULD APPROCACH
        ## THE INLET AS CLOSE AS POSSIBLE TO GET THE HIGHEST VELOCITY.
        ## THIS IS A BAD STRATEGY.
        return K_RWD*np.dot(vec1, vec2)
      '''      
            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,action):
        self.it += 1
        done   = False
        reach  = False

        id_non0 = np.nonzero(self.reach_targets)
        if id_non0[0].size == 0:
            id_tar=0;
        else:
            id_tar=id_non0[0][-1] + 1
        ## the id_tar th target to reach  from 0 to self.ntarges-1  

        x_tar = self.x_tar[:,id_tar]
        
        state_old=self.state
        action=action.T
        #print('action')
        state_n = mf.step_lbm(len(action), action, self.dt, state_old)
        self.state_n = state_n 
        print(state_n)
        
        '''
        reward_orient1 = self.cal_reward(self.state[:self.nparticles], self.state_n[:self.nparticles], x_tar[:self.nparticles])
        reward_orient2 = self.cal_reward(self.state[self.nparticles:], self.state_n[self.nparticles:], x_tar[self.nparticles:])
        '''
#        reward_action = COEF_EXCEED*reward_exceed(q, self.max_rate, -self.max_rate)
        reward = 0.0
        
        
        if np.linalg.norm(self.state_n[:self.nparticles]-self.state_n[self.nparticles:])<0.21:
            done = True
            reward += 0            
        
        
        '''
        ind_min_in1, d_min_in1 = self.min_dist_2_inlets(self.state_n[:self.nparticles], self.rinlets)
        ind_min_in2, d_min_in2 = self.min_dist_2_inlets(self.state_n[self.nparticles:], self.rinlets)
        
        if min(d_min_in1,d_min_in2) < DIST_EPSI_INLET: #exit the inlet
            done = True
            reward += 0
        '''  
        if np.linalg.norm(self.state_n[:2])>0.9  or np.linalg.norm(self.state_n[2:]) >0.9:
            done = True
             

        #print (self.state_n)
        #print (x_tar)
        if np.linalg.norm(self.state_n[:2]-x_tar[:2])+ np.linalg.norm(self.state_n[2:]-x_tar[2:])< 0.23 and np.linalg.norm(self.state_n[:self.nparticles]-self.state_n[self.nparticles:])<0.21:
        #if np.linalg.norm((self.state_n[:2]+self.state_n[2:])/2.0-x_tar) < self.dist_epsi_goal and self.it>50:            
            
            self.reach_targets[id_tar] = 1
            #print('herelll')
            #print(self.reach_targets)
            #print([id_tar, self.ntargets-1])
            reward += (id_tar+1)*0
            if id_tar == self.ntargets-1:
                done = True                
                #print(done)
                
        self.state = self.state_n
        
        if self.it>=MAX_STEP:
            done = True
            
        return self._get_obs(), reward, done, {}

    def reset(self):
        self.it=0
        self.state = np.array([X_ini, Y_ini,X_ini2, Y_ini2],dtype=np.float32)
        self.reach_targets = np.zeros(self.ntargets, dtype=int)        
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate((self.state, self.reach_targets)) 
    
    def render(self):
        pass
        return 

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



