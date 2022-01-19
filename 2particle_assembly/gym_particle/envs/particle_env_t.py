import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math
import numpy.matlib
import main as mf
from numpy import linalg as LA

num_inlets = 3
K_RWD = 4;
RWD_TAR = 1000 # reward to gain when the point approaches the target.
RWD_OUT = -500 # rewad when the particle get out of the domain.
DIST_EPSI_GOAL = 0.05
COEF_EXCEED=-0.1
DIST_EPSI_INLET=0.05 # need to prevent the particle from too close to the inlets.
MAX_STEP=50000
DT = 0.02
goal_step=200
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
ACTION_MEAN = (ACTION_LOW + ACTION_HIGH)/2

X_ini = 0.6
Y_ini = 0.2

X_tar = [-0.8]
Y_tar = [0.2]

r_tar=1


class ParticleEnv_t(gym.Env):
    metadata = {
        'render.modes' : ['human'],
        'video.frames_per_second' : 30
    }

    def __init__(self, num_inlets=num_inlets):
        self.ntargets = len(X_tar)
        self.max_rate = 1
        self.xmax = 1
        self.n = num_inlets
        self.dt=DT
        self.viewer = None
        nobs = 2 + self.ntargets
        self.action_space = spaces.Box(low=-self.max_rate, high=self.max_rate, shape=(self.n-1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.xmax, high=self.xmax, shape=(nobs,), dtype=np.float32)
        self.X_ini = X_ini
        self.Y_ini = Y_ini
        self.X_tar = X_tar
        self.Y_tar = Y_tar
        self.r_tar = r_tar
        self.dist_epsi_goal = DIST_EPSI_GOAL
        self.ACTION_LOW = ACTION_LOW
        self.ACTION_HIGH = ACTION_HIGH
        ACTION_MEAN = (self.ACTION_LOW + self.ACTION_HIGH)/2.
        self.rinlets = np.zeros((self.n,2),dtype=np.float32)
        self.seed()
        self.reward=0
        self.nstep=5
        self.x_tar = np.array([self.X_tar, self.Y_tar], dtype=np.float32) ## 2 rows * ntargets columns
#        print('herer')
#        print(self.x_tar)
        self.goal_step=goal_step
        self.reach_targets = np.zeros(self.ntargets, dtype=int)
        self.it = 0

            
            
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
        state_old = self.state
        action=action.T
        #print('action')
        self.state_n = mf.step_lbm(len(action), action, self.dt, state_old)        
        #print(self.state_n)
        self.reward+=LA.norm(self.state_n-self.state)
        '''
        reward_step = -1
        reward_orient = self.cal_reward(self.state, self.state_n, x_tar)
        '''
#        reward_action = COEF_EXCEED*reward_exceed(q, self.max_rate, -self.max_rate)
#        reward = reward_orient + reward_step


#            reward += RWD_OUT
            
        # if np.linalg.norm(self.state_n)>1:
        #     done = True
        #     reward += RWD_OUT

#        print (self.state_n)
#        print (x_tar)
#if np.linalg.norm(self.state_n-x_tar) < self.dist_epsi_goal and abs(self.reward-r_tar)<self.dist_epsi_goal:
        if self.it>self.goal_step and np.linalg.norm(self.state_n-x_tar) < self.dist_epsi_goal:
            self.reach_targets[id_tar] = 1
#            print('herelll')
#            print(self.reach_targets)
#            print([id_tar, self.ntargets-1])
#            reward += (id_tar+1)*RWD_TAR
            if id_tar == self.ntargets-1:
                done = True                
#                print(done)
                
        self.state = self.state_n
        
        if self.it>=MAX_STEP:
            done = True
            
        return self._get_obs(), self.reward, done, {}

    def reset(self):
        self.it=0
        self.state = np.array([X_ini, Y_ini],dtype=np.float32)
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


