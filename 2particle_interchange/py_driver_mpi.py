"""
Same as approximate dynamics, but now the input is sine and cosine of theta (output is still dtheta)
This is a continuous representation of theta, which some papers show is easier for a NN to learn.
"""
import os
import sys
import shutil
import gym
import numpy as np
import torch
import math
import main as mf
from mpi4py import MPI
import gym_particle
import pppi_trap_ntar2
from gym import wrappers, logger as gym_log
from datetime import datetime
#from array import array
import matplotlib
import matplotlib.pyplot as plt
import random



# logging.basicConfig(level=logging.INFO,
#                     format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
#                     datefmt='%m-%d %H:%M:%S')

DT = 0.05
dt=0.01

N_INLETS = 6

'''
X_ini=0.7*torch.rand(1)*math.cos(2*math.pi*torch.rand(1))
Y_ini= 0.7*torch.rand(1)*math.sin(2*math.pi*torch.rand(1))
X_ini2= 0.7*torch.rand(1)*math.cos(2*math.pi*torch.rand(1))
Y_ini2= 0.7*torch.rand(1)*math.sin(2*math.pi*torch.rand(1))
'''
'''
X_ini=-0.53
Y_ini=0.58
X_ini2=0
Y_ini2=0.83
'''

'''
X_ini =0.5
Y_ini =0
X_ini2 =-X_ini
Y_ini2 =-Y_ini
'''
'''
X_ini=0.5
Y_ini=0.7
X_ini2=-0.8
Y_ini2=-0.3
'''

'''
print(X_ini)
print(Y_ini)
print(X_ini2)
print(Y_ini2)
X_tar = [X_ini2]
Y_tar = [Y_ini2]
X_tar2 = [X_ini]
Y_tar2 = [Y_ini]
'''
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
NV=5000
NPARTICLES=2
N_EPISODES = 1000
BOOT_STRAP_ITER = 10000
TRAIN_EPOCH = 500
lr=0.01
TIMESTEPS = 1 # T %%INCREASE THIS VALUE SEEMS TO BE IMPORTANT
N_SAMPLES = 100 # K seems to be able to get out trap using a small value
ACTION_SIG = 1.0  ## seems to be best when it is order of action_low and action_high
MIN_X = -0.8
MAX_X =  0.8
MIN_Y = -0.8
MAX_Y =  0.8
MINVX = -3.0
MINVY =-3.0
MAXVX =3.0
MAXVY = 3.0
DIST_EPSI_GOAL = 0.01
root_rank = 0
ENV_NAME = "particle-v0"
comm = MPI.COMM_WORLD

fcomm = comm.py2f()
mf.init_lbm(fcomm)

rank = comm.Get_rank()
if __name__ == "__main__":
    nx = 4
    nu = N_INLETS - 1
    nnx= 2
#    N_tar = len(X_tar)
    N_tar =1

    
    
    rinlets = torch.zeros(N_INLETS,2)

#    POSITION_INI = torch.Tensor([X_ini, Y_ini,X_ini2, Y_ini2]).t()
#    POSITION_TAR = torch.Tensor([X_tar, Y_tar,X_tar2, Y_tar2]).t()
    
    NV=5000
    NPARTICLES=2
    N_EPISODES = 1000
    BOOT_STRAP_ITER = 10000
    TRAIN_EPOCH = 500
    lr=0.01
    TIMESTEPS = 1 # T %%INCREASE THIS VALUE SEEMS TO BE IMPORTANT
    N_SAMPLES = 1000 # K seems to be able to get out trap using a small value
    ACTION_SIG = 1.0  ## seems to be best when it is order of action_low and action_high
    MIN_X = -0.8
    MAX_X =  0.8
    MIN_Y = -0.8
    MAX_Y =  0.8
    MINVX = -3.0
    MINVY =-3.0
    MAXVX =3.0
    MAXVY = 3.0
    if rank==root_rank:
        now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
        traj_path = os.path.join(os.getcwd(), 'traj_data', ENV_NAME, now)
        if not os.path.exists(traj_path):
            os.makedirs(traj_path)
#    LOG_FILENAME = os.path.join(traj_path, "log")
#    logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO,format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
#                        datefmt='%m-%d %H:%M:%S')
    
        cur_py = os.path.join(os.getcwd() ,__file__)
        shutil.copy(cur_py, traj_path)
    else:
        pass
    

    WEIGHT_POSITION = 100
    WEIGHT_DIST = 100
    WEIGHT_ANGLE = 30
    WEIGHT_ACTION = 0
    angle=math.pi/6.0
#    C_OUTBOUND = 1e7
    H_UNITS = 90
#    NV = 1000
    N_ITERATIONS = 10000 
    ZERO_AFTER_RETRAIN = False
    
    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double

    #noise_sigma = torch.tensor(1, device=d, dtype=dtype)
#    noise_sigma = torch.tensor([[ACTION_SIG,0,0,0,0],[0,ACTION_SIG,0,0,0],[0,0,ACTION_SIG,0,0],[0,0,0,ACTION_SIG,0], [0,0,0,0, ACTION_SIG]], device=d, dtype=dtype) ## shape nu * nu
    a=ACTION_SIG*torch.ones((N_INLETS-1),device=d, dtype=dtype)
    noise_sigma =torch.diag(a)
#    print(noise_sigma )
#    noise_sigma = torch.tensor([[ACTION_SIG,0,0,0,0,0,0],[0,ACTION_SIG,0,0,0,0,0],[0,0,ACTION_SIG,0,0,0,0],[0,0,0,ACTION_SIG,0,0,0], [0,0,0,0, ACTION_SIG,0,0],[0,0,0,0,0, ACTION_SIG,0],[0,0,0,0,0,0, ACTION_SIG]], device=d, dtype=dtype) ## shape nu * n
    ## if the actions are independent of each other, sigma_{12} = sigma_{21} = 0 
    lambda_ = 1
    hunits=100
    network = torch.nn.Sequential(
        torch.nn.Linear(9, hunits),
        torch.nn.Tanh(),
        torch.nn.Linear(hunits,hunits),
        torch.nn.Tanh(),       
        torch.nn.Linear(hunits, 4),
    ).double().to(device=d)
    network.load_state_dict(torch.load('network_lbm63.pth')) 

            

    def tanh_limit(action, act_l, act_h):
        act_m = (act_l + act_h)/2
        return act_m + (act_h - act_l)/2 * torch.tanh(action - act_m)
            
    def dynamics(state, perturbed_action):
        q = tanh_limit(perturbed_action, ACTION_LOW, ACTION_HIGH)
        #q = torch.clamp(perturbed_action, ACTION_LOW, ACTION_HIGH)
        if state.dim() is 1 or q.dim() is 1:
            state = state.view(1, -1) # data of one row
            q = q.view(1, -1)
        statexy=state
        q_n = transform(q,ACTION_LOW,ACTION_HIGH)
        for i in range(5):
            xu = torch.cat((statexy, q_n), dim=1) ## cat along the 1 dimension (0: along column direction; 1: along row direction)
            #print(xu.shape)
            k1 = dt*rever_transform(network(xu).detach(),MINVX,MAXVX)
            xu2=torch.cat((statexy+0.5*k1, q), dim=1)
            k2 = dt*rever_transform(network(xu2).detach(),MINVX,MAXVX)
            statexy=statexy+k2
            if i==0:
                K=k2
            else:
                K+=k2
        return [statexy, K]
    

    

    def true_dynamics(state, perturbed_action):
        pass
    
    def transform(input_,min_value,max_value):
        input_=2.0*(input_-min_value)/(max_value-min_value)-1.0
        return input_
    
    def rever_transform(input_,min_value,max_value):
        input_=(0.5*(input_+1.0)*(max_value-min_value)+min_value)
        return input_    
      



    def running_cost(state,action, prev_state,dstate,reach_targets):

        position = state; # K * nx
        # dstate K * nx; prev_state 1 * nx
        prev_position = prev_state
        
        id_non0 = np.nonzero(reach_targets)
        if id_non0[0].size == 0:
            id_tar = 0;
        else:
            id_tar = id_non0[0][-1] + 1
           
        current_tar = POSITION_TAR[id_tar, :]
        vec2 = current_tar - position

#        print(reach_targets)

        vec1 = dstate
        vec2 = current_tar -prev_position 
#        vec2 = current_tar - position
        vec21=torch.cat(((vec2[:,0]*math.cos(angle)-vec2[:,1]*math.sin(angle)).view(-1,1),(vec2[:,0]*math.sin(angle)+vec2[:,1]*math.cos(angle)).view(-1,1)),dim=1)
        vec22=torch.cat(((vec2[:,2]*math.cos(angle)-vec2[:,3]*math.sin(angle)).view(-1,1),(vec2[:,2]*math.sin(angle)+vec2[:,3]*math.cos(angle)).view(-1,1)),dim=1)
        vec2=torch.cat((vec21.view(1,-1),vec22.view(1,-1)),dim=1).view(1,-1).repeat(position.shape[0],1) 





        costhe = torch.sum(torch.mul(vec1, vec2), 1) / (torch.norm(vec1, dim=1) * torch.norm(vec2, dim=1)) ##cannot neglect dim here
        the = torch.acos(costhe)
#        cost_the = 1 - costhe ** 2 ## FAIL
        cost_the = 1 - costhe ## works well

        dist_circle = torch.sum(position ** 2, dim=1) - 0.25
#        print("dist")
#       print(dist_circle)
        dist_circle[dist_circle < 0] = 0

        prev_dist_sqr = torch.sum ((prev_position - current_tar)**2  , dim=1)
        #cost = WEIGHT_ANGLE * cost_the
        
#        cost = WEIGHT_POSITION * torch.sum(((position - current_tar) ** 2) , dim=1)  + WEIGHT_ACTION * torch.sum(action ** 2, dim=1)  + WEIGHT_ANGLE * cost_the
        cost1=torch.sum(((position[:,:2] - position[:,2:])**2 ) , dim=1)
        dist=torch.sum(((abs(position[:,:2] - position[:,2:])-0.2)**2 ) , dim=1)

        cost = WEIGHT_POSITION * torch.sum(((position - current_tar)**2 ) , dim=1) / prev_dist_sqr + WEIGHT_ACTION * torch.sum(action ** 2, dim=1) + WEIGHT_ANGLE * cost_the

        return cost            

        
    def retrain(new_data):
        pass

    

    

        




#    env.state = np.array([X_ini, Y_ini,X_ini2, Y_ini2])
    for e in range(1):
#        print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
#        print(e)

        X_ini=-0.5
        Y_ini=-0.5
        X_ini2=0.5
        Y_ini2=0.5

    
        X_tar = [X_ini2]
        Y_tar = [Y_ini2]
        X_tar2 = [X_ini]
        Y_tar2 = [Y_ini]

        
        POSITION_INI = torch.Tensor([X_ini, Y_ini,X_ini2, Y_ini2]).t()
        POSITION_TAR = torch.Tensor([X_tar, Y_tar,X_tar2, Y_tar2]).t()



        env = gym.make(ENV_NAME)  # bypass the default TimeLimit wrapper

        env.reset()
        env.state = np.array([X_ini, Y_ini,X_ini2, Y_ini2])
        ##overwrite environment variables.
        env.dt = DT
        env.n = N_INLETS
        env.X_ini = X_ini
        env.Y_ini = Y_ini
        env.X_tar = X_tar
        env.Y_tar = Y_tar
        env.X_ini2 = X_ini2
        env.Y_ini2 = Y_ini2
        env.X_tar2 = X_tar2
        env.Y_tar2 = Y_tar2
        env.x_tar = np.array([X_tar, Y_tar,X_tar2, Y_tar2], dtype=np.float32) ## 2 rows * ntargets columns
        env.dist_epsi_goal = DIST_EPSI_GOAL
        env.ACTION_LOW = ACTION_LOW
        env.ACTION_HIGH = ACTION_HIGH
        env.state = np.array([X_ini, Y_ini,X_ini2, Y_ini2])      
        if rank==root_rank:
            mppi_gym = pppi_trap_ntar2.MPPI(dynamics, true_dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                             lambda_=lambda_, device=d, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
                             u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))
        else:
            mppi_gym = None
        mppi_gym = comm.bcast(mppi_gym, root=root_rank)                    
        total_reward, _, traj, done, iter_act, r_targets = pppi_trap_ntar2.run_mppi(mppi_gym, env,rank,root_rank,comm,retrain ,ifzero = ZERO_AFTER_RETRAIN, iter = N_ITERATIONS)
        if rank==root_rank:        
            n_targets_reach = np.nonzero(r_targets)[0].size
            if n_targets_reach == N_tar:
                succeed = True           
            else:
                succeed = False        
                print(e, total_reward,done, succeed, n_targets_reach, iter_act)

            
            traj_bin = os.path.join(traj_path,"trajectory_"+str(e)+".bin")
            n_record_per_traj = os.path.join(traj_path,"n_records_"+str(e)+".bin")
            ini_tar_bin = os.path.join(traj_path,"ini_tar_"+str(e)+".bin")
            np.array(traj.shape[1]).astype('int').tofile(n_record_per_traj)        
            traj.astype('float').tofile(traj_bin)
            
            ini_tar = np.array(([N_tar] + [X_ini] + X_tar + [Y_ini] + Y_tar+[X_ini2] + X_tar2 + [Y_ini2] + Y_tar2))
            ini_tar.astype('float').tofile(ini_tar_bin)
        else:
            pass
