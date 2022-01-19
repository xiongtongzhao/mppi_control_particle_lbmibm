import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math
import numpy.matlib
from numpy import linalg as LA
import main as mf
'''
from  path2_gym_gym import tang_N1 as tang_N1
from  path2_gym_gym import tang_N2 as tang_N2
from  path2_gym_gym import tang_N3 as tang_N3

from  path2_gym_gym import tang_U1 as tang_U1
from  path2_gym_gym import tang_U2 as tang_U2
from  path2_gym_gym import tang_U3 as tang_U3

from  path2_gym_gym import tang_S1 as tang_S1
from  path2_gym_gym import tang_S2 as tang_S2
from  path2_gym_gym import tang_S3 as tang_S3
'''
'''
from  path2_gym import tang_N1 as tang_N1
from  path2_gym import tang_N2 as tang_N2
from  path2_gym import tang_N3 as tang_N3

from  path2_gym import tang_NN1 as tang_U1
from  path2_gym import tang_NN2 as tang_U2
from  path2_gym import tang_NN3 as tang_U3

from  path2_gym import tang_NNN1 as tang_S1
from  path2_gym import tang_NNN2 as tang_S2
from  path2_gym import tang_NNN3 as tang_S3
'''
from path2_gym import char_N1 as char_N1
from path2_gym import char_N2 as char_N2
from path2_gym import char_N3 as char_N3

from path2_gym import char_U1 as char_U1
from path2_gym import char_U2 as char_U2
from path2_gym import char_U3 as char_U3

from path2_gym import char_S1 as char_S1
from path2_gym import char_S2 as char_S2
from path2_gym import char_S3 as char_S3

from path2_gym import tang_N1 as tang_N1
from path2_gym import tang_N2 as tang_N2
from path2_gym import tang_N3 as tang_N3

from path2_gym import tang_U1 as tang_U1
from path2_gym import tang_U2 as tang_U2
from path2_gym import tang_U3 as tang_U3

from  path2_gym import tang_S1 as tang_S1
from  path2_gym import tang_S2 as tang_S2
from  path2_gym import tang_S3 as tang_S3

from  path2_gym import closest_N1 as closest_N1
from  path2_gym import closest_N2 as closest_N2
from  path2_gym import closest_N3 as closest_N3

from  path2_gym import closest_U1 as closest_U1
from  path2_gym import closest_U2 as closest_U2
from  path2_gym import closest_U3 as closest_U3

from  path2_gym import closest_S1 as closest_S1
from  path2_gym import closest_S2 as closest_S2
from  path2_gym import closest_S3 as closest_S3


num_particles=3
num_inlets = 8
K_RWD = 4;
RWD_TAR = 1000 # reward to gain when the point approaches the target.
RWD_OUT = -500 # rewad when the particle get out of the domain.
DIST_EPSI_GOAL = 0.004
COEF_EXCEED=-0.1
DIST_EPSI_INLET=0.05 # need to prevent the particle from too close to the inlets.
MAX_STEP=20000
DT = 0.02
goal_step=50
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
ACTION_MEAN = (ACTION_LOW + ACTION_HIGH)/2
MINVX = -4.0
MINVY =-2.0
MAXVX =4.0
MAXVY = 2.0

X_ini1 = 0.6
X_ini2 =0.6
X_ini3 = 0.6
Y_ini1 = 0.2
Y_ini2 =0.2
Y_ini3=0.2
X_tar1 = [-0.8,0.0,0.0]
Y_tar1 = [0.2,0.0,0.0]
X_tar2 =[-0.8 ,0.0,0.0]
Y_tar2 = [-0.8,0.0,0.0 ]
X_tar3 =[-0.8 ,0.0,0.0]
Y_tar3 = [-0.8,0.0,0.0 ]





class ParticleEnv(gym.Env):
    metadata = {
        'render.modes' : ['human'],
        'video.frames_per_second' : 30
    }

    def __init__(self, num_inlets=num_inlets):
        self.ntargets1 = len(X_tar1)
        self.ntargets2 = len(X_tar2)
        self.ntargets3 = len(X_tar3)        
        self.max_rate = 1
        self.xmax = 1
        self.nparticles=num_particles
        self.n = num_inlets
        self.dt=DT
        self.viewer = None
        nobs = 2 + self.ntargets1
        self.action_space = spaces.Box(low=-self.max_rate, high=self.max_rate, shape=(self.n-1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.xmax, high=self.xmax, shape=(nobs,), dtype=np.float32)
        self.X_ini1 = X_ini1
        self.Y_ini1 = Y_ini1
        self.X_tar1 = X_tar1
        self.Y_tar1 = Y_tar1
        self.X_ini2 = X_ini2
        self.Y_ini2 = Y_ini2
        self.X_tar2 = X_tar2
        self.Y_tar2 = Y_tar2
        self.X_ini3 = X_ini3
        self.Y_ini3 = Y_ini3
        self.X_tar3 = X_tar3
        self.Y_tar3 = Y_tar3       
        self.reward1= 0.0    
        self.reward2 = 0.0        
        self.reward3 = 0.0        
        self.reward = 0
        self.closest1=np.array([0.0, 0.0], dtype=np.float32)
        self.closest2=np.array([0.0, 0.0], dtype=np.float32)       
        self.closest3=np.array([0.0, 0.0], dtype=np.float32)       
        
        self.dist_epsi_goal = DIST_EPSI_GOAL
        self.ACTION_LOW = ACTION_LOW
        self.ACTION_HIGH = ACTION_HIGH
        ACTION_MEAN = (self.ACTION_LOW + self.ACTION_HIGH)/2.
        self.rinlets = np.zeros((self.n,2),dtype=np.float32)
        self.seed()
        
        self.x_tar1 = np.array([self.X_tar1, self.Y_tar1], dtype=np.float32) ## 2 rows * ntargets columns
        #print('herer')
        self.x_tar2 = np.array([self.X_tar2, self.Y_tar2], dtype=np.float32) 
        #print(self.x_tar)
        self.x_tar3 = np.array([self.X_tar3, self.Y_tar3], dtype=np.float32)
        self.reach_targets1 = np.zeros(self.ntargets1, dtype=int)
        self.reach_targets2 = np.zeros(self.ntargets2, dtype=int)
        self.reach_targets3 = np.zeros(self.ntargets3, dtype=int)       
        self.it = 0
        phi = 2*math.pi/self.n
        for i in range(self.n):
            angle = i*phi
            self.rinlets[i,:] = np.array([math.cos(angle), math.sin(angle)],dtype=np.float32)
        self.dd=0
    def min_dist_2_inlets(self, state, rinlets):
        state_ = np.matlib.repmat(state,self.n,1)
        r      = state_ - rinlets
        d      = np.squeeze(np.linalg.norm(r,axis=1))
        ind    = np.argmin(d)
        return ind, d[ind]  ## index of the inlets cloest to the current position. 
    
    def cal_reward(self, state, state_next, target):
        vec1 = np.squeeze(state_next - state)/self.dt
        vec2 = np.squeeze(target - state)
        vec2 = vec2/np.linalg.norm(vec2)
        ## THIS TENDS TO MAXIMIZE THE VELOCITY AND HENCE THE PARTICLE WOULD APPROCACH
        ## THE INLET AS CLOSE AS POSSIBLE TO GET THE HIGHEST VELOCITY.
        ## THIS IS A BAD STRATEGY.
        return K_RWD*np.dot(vec1, vec2)
    

            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,action):
        self.it += 1
        done   = False
        reach  = False

        id_non1 = np.nonzero(self.reach_targets1)
        if id_non1[0].size == 0:
            id_tar1=0;
        else:
            id_tar1=id_non1[0][-1] + 1
        ## the id_tar th target to reach  from 0 to self.ntarges-1  
        id_non2 = np.nonzero(self.reach_targets2)
        if id_non2[0].size == 0:
            id_tar2=0;
        else:
            id_tar2=id_non2[0][-1] + 1
        id_non3 = np.nonzero(self.reach_targets3)
        if id_non3[0].size == 0:
            id_tar3=0;
        else:
            id_tar3=id_non3[0][-1] + 1            
        
        x_tar1 = self.x_tar1[:,id_tar1]
        x_tar2 = self.x_tar2[:,id_tar2]
        x_tar3 = self.x_tar3[:,id_tar3]
        state_old=self.state
        action=action.T
        #print('action')
        state_n = mf.step_lbm(len(action), action, self.dt, state_old)
        self.state_n = state_n        
        
        
        
    
        #print(u)
        #print(self.state)
        #print(self.reward1)
        if id_tar1==0:
            self.reward1+=-np.dot((self.closest1-closest_N1(self.state[:2])),tang_N1(self.closest1))/LA.norm(tang_N1(self.closest1))
            self.closest1=closest_N1(self.state[:2])
            #print(char_N1(self.closest1))             
            
        elif id_tar1==1:

            self.reward1+=-np.dot((self.closest1-closest_N2(self.state[:2])),tang_N2(self.closest1))/LA.norm(tang_N2(self.closest1))
            self.closest1=closest_N2(self.state[:2])
            #print(char_N2(self.closest1))            
        else:
           
            self.reward1+=-np.dot((self.closest1-closest_N3(self.state[:2])),tang_N3(self.closest1))/LA.norm(tang_N3(self.closest1))
            self.closest1=closest_N3(self.state[:2])              
            #print(char_N3(self.closest1)) 
        if id_tar2==0:
            self.reward2+=-np.dot((self.closest2-closest_U1(self.state[2:4])),tang_U1(self.closest2))/LA.norm(tang_U1(self.closest2))       
            self.closest2=closest_U1(self.state[2:4])              
            #print(char_U1(self.closest2)) 
        elif id_tar2==1:
            self.reward2+=-np.dot((self.closest2-closest_U2(self.state[2:4])),tang_U2(self.closest2))/LA.norm(tang_U2(self.closest2))             
            self.closest2=closest_U2(self.state[2:4])            
            #print(char_U2(self.closest2))             
        else:
            self.reward2+=-np.dot((self.closest2-closest_U3(self.state[2:4])),tang_U3(self.closest2))/LA.norm(tang_U3(self.closest2))          
            self.closest2=closest_U3(self.state[2:4])            
            #print(char_U3(self.closest2))             
        if id_tar3==0:
            #print(closest_S1(self.state[4:]))
            self.reward3+=-np.dot((self.closest3-closest_S1(self.state[4:])),tang_S1(self.closest3))/LA.norm(tang_S1(self.closest3))               
            self.closest3=closest_S1(self.state[4:])
            #print(tang_S1(self.closest3)) 
        elif id_tar3==1:
            self.reward3+=-np.dot((self.closest3-closest_S2(self.state[4:])),tang_S2(self.closest3))/LA.norm(tang_S2(self.closest3))           
            self.closest3=closest_S2(self.state[4:])            
            #print(char_S2(self.closest3)) 
        else:
          
            self.reward3+=-np.dot((self.closest3-closest_S3(self.state[4:])),tang_S3(self.closest3))/LA.norm(tang_S3(self.closest3))
            self.closest3=closest_S3(self.state[4:])            
            #print(char_S3(self.closest3))   
           

        '''
        reward_orient1 = self.cal_reward(self.state[:self.nparticles], self.state_n[:self.nparticles], x_tar[:self.nparticles])
        reward_orient2 = self.cal_reward(self.state[self.nparticles:], self.state_n[self.nparticles:], x_tar[self.nparticles:])
        '''
#        reward_action = COEF_EXCEED*reward_exceed(q, self.max_rate, -self.max_rate)


        '''
        self.reward1+=LA.norm(u1*self.dt)
        self.reward2+=LA.norm(u2*self.dt)
        self.reward3+=LA.norm(u3*self.dt)
        '''
        #print(reward_step1/LA.norm(u1*self.dt))
        '''
        ind_min_in1, d_min_in1 = self.min_dist_2_inlets(self.state_n[:2], self.rinlets)
        ind_min_in2, d_min_in2 = self.min_dist_2_inlets(self.state_n[2:-2], self.rinlets)
        ind_min_in3, d_min_in3 = self.min_dist_2_inlets(self.state_n[-2:], self.rinlets)
        if min(d_min_in1,d_min_in2,d_min_in3) < DIST_EPSI_INLET: #exit the inlet
            done = True
        '''
        # if np.linalg.norm(self.state_n)>1:
        #     done = True
        #     reward += RWD_OUT

        #print (self.state_n)
        #print (x_tar)
        if np.linalg.norm(self.closest1-x_tar1) < self.dist_epsi_goal:
            self.reach_targets1[id_tar1] = 1
            #print('herelll')
            #print(self.reach_targets)
            #print([id_tar, self.ntargets-1])

            if id_tar1 == self.ntargets1-1:
                done = True                
                #print(done)
        
        if np.linalg.norm(self.closest2-x_tar2) < self.dist_epsi_goal:
            self.reach_targets2[id_tar2] = 1

            if id_tar2== self.ntargets2-1:
                done = True        
       
        if np.linalg.norm(self.closest3-x_tar3) < self.dist_epsi_goal:
            self.reach_targets3[id_tar3] = 1

            if id_tar3== self.ntargets3-1:
                done = True
                
        if np.linalg.norm(self.state_n[:2]-self.state_n[2:4])<0.2:
            done = True
            
        if np.linalg.norm(self.state_n[:2]-self.state_n[4:])<0.2:
            done = True
            
        if np.linalg.norm(self.state_n[4:]-self.state_n[2:4])<0.2:
            done = True
            
        if np.linalg.norm(self.state_n[:2])>0.9  or np.linalg.norm(self.state_n[2:4])  >0.9 or np.linalg.norm(self.state_n[4:])  >0.9:
             done = True              
       
        
        
        self.state = self.state_n
        
        if self.it>=MAX_STEP:
            done = True
            
        return self._get_obs(), self.reward1, self.reward2, self.reward3,self.closest1,self.closest2,self.closest3,done, {}

    def reset(self):
        self.it=0
        self.state = np.array([X_ini1, Y_ini1,X_ini2, Y_ini2,X_ini3, Y_ini3],dtype=np.float32)
        self.reach_targets1 = np.zeros(self.ntargets1, dtype=int)
        self.reach_targets2 = np.zeros(self.ntargets2, dtype=int)
        self.reach_targets3 = np.zeros(self.ntargets3, dtype=int)        
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate((self.state, self.reach_targets1,self.reach_targets2,self.reach_targets3)) 
    
    def render(self):
        pass
        return 

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



