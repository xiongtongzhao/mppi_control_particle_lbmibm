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
import logging
import math
import main as mf
from mpi4py import MPI
#sys.path.insert(0, '../mppi')
import nppi_trap_ntar3t_NN #import the mppi_NN 
from gym import wrappers, logger as gym_log
from datetime import datetime
#from array import array
import matplotlib as mpl
import matplotlib.pyplot as plt
import gym_particle
from path.path2_main import char_N1 as char_N1
from path.path2_main import char_N2 as char_N2
from path.path2_main import char_N3 as char_N3
from path.path2_main import tang_N1 as tang_N1
from path.path2_main import tang_N2 as tang_N2
from path.path2_main import tang_N3 as tang_N3
from path.path2_main import char_U1 as char_U1
from path.path2_main import char_U2 as char_U2
from path.path2_main import char_U3 as char_U3
from path.path2_main import tang_U1 as tang_U1
from path.path2_main import tang_U2 as tang_U2
from path.path2_main import tang_U3 as tang_U3
from path.path2_main import char_S1 as char_S1
from path.path2_main import char_S2 as char_S2
from path.path2_main import char_S3 as char_S3
from path.path2_main import tang_S1 as tang_S1
from path.path2_main import tang_S2 as tang_S2
from path.path2_main import tang_S3 as tang_S3





# logging.basicConfig(level=logging.INFO,
#                     format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
#                    datefmt='%m-%d %H:%M:%S')
R=0.2
L=0.15
SN=4*R+2*math.sqrt(2)*R
SU=2*R+math.pi*R
SS=1.5*(math.pi+2)*R


DT = 0.05
dt=0.01
N_INLETS = 8


ACTION_LOW = -1
ACTION_HIGH = 1
MIN_X = -0.8
MAX_X =  0.8
MIN_Y = -0.8
MAX_Y =  0.8
MINVX = -4
MINVY =-2
MAXVX =4
MAXVY = 2
r=0.8


DIST_EPSI_GOAL = 0.012


root_rank = 0
ENV_NAME = "particle-v0"
comm = MPI.COMM_WORLD

fcomm = comm.py2f()
mf.init_lbm(fcomm)

rank = comm.Get_rank()
if __name__ == "__main__":
    nx = 6
    nu = N_INLETS - 1
    nnx= 2
#    N_tar = len(X_tar)
    if rank==root_rank:
        now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
        traj_path = os.path.join(os.getcwd(), 'traj_data', ENV_NAME, now)
        if not os.path.exists(traj_path):
            os.makedirs(traj_path)

    
        cur_py = os.path.join(os.getcwd() ,__file__)
        shutil.copy(cur_py, traj_path)
    else:
        pass
    
    
    rinlets = torch.zeros(N_INLETS,2)

#    POSITION_INI = torch.Tensor([X_ini, Y_ini,X_ini2, Y_ini2]).t()
#    POSITION_TAR = torch.Tensor([X_tar, Y_tar,X_tar2, Y_tar2]).t()
    
    NV=5000
    NPARTICLES=2
    N_EPISODES = 1
    BOOT_STRAP_ITER = 10000
    TRAIN_EPOCH = 500
    lr=0.01
    TIMESTEPS = 1 # T %%INCREASE THIS VALUE SEEMS TO BE IMPORTANT
    N_SAMPLES = 50000 # K seems to be able to get out trap using a small value
    ACTION_SIG = 1  ## seems to be best when it is order of action_low and action_high
    MIN_X = -1
    MAX_X =  1
    MIN_Y = -1
    MAX_Y =  1
    ERROR1=[]
    ERROR2=[]
    WEIGHT_POSITION = 4300
    WEIGHT_VELOCITY = 0
    WEIGHT_ANGLE = 30
    WEIGHT_ANGLE2 = 0
    WEIGHT_ACTION = 0
    powern=1
#    C_OUTBOUND = 1e7
    hunits = 100
#    NV = 1000
    N_ITERATIONS = 1000 
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
    p=0.0

    network = torch.nn.Sequential(
        torch.nn.Linear(13, hunits),
        torch.nn.Tanh(),
        torch.nn.Linear(hunits,hunits),
        torch.nn.Tanh(),
        torch.nn.Linear(hunits,hunits),
        torch.nn.Tanh(),       
        torch.nn.Linear(hunits, 6),
    ).double().to(device=d)
    network.load_state_dict(torch.load('network_lbm8.pth'))
    
    
    phi = 2*math.pi / N_INLETS
    for i in range(N_INLETS):
        angle = i*phi
        rinlets[i,:] = torch.from_numpy(np.array([math.cos(angle), math.sin(angle)], dtype=np.float32))
            

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
        q_n = q
        for i in range(5):
            xu = torch.cat((statexy, q_n), dim=1) ## cat along the 1 dimension (0: along column direction; 1: along row direction)
            #print(xu.shape)
            U=network(xu).detach()

            U[:,0]=rever_transform(U[:,0],MINVX,MAXVX)
            U[:,1]=rever_transform(U[:,1],MINVY,MAXVY)           
            U[:,2]=rever_transform(U[:,2],MINVX,MAXVX)
            U[:,3]=rever_transform(U[:,3],MINVY,MAXVY)
            U[:,4]=rever_transform(U[:,4],MINVX,MAXVX)
            U[:,5]=rever_transform(U[:,5],MINVY,MAXVY)            
            k1 = dt*U
            xu2=torch.cat((statexy+0.5*k1, q), dim=1)
            U=network(xu2).detach()
        
            U[:,0]=rever_transform(U[:,0],MINVX,MAXVX)
            U[:,1]=rever_transform(U[:,1],MINVY,MAXVY)           
            U[:,2]=rever_transform(U[:,2],MINVX,MAXVX)
            U[:,3]=rever_transform(U[:,3],MINVY,MAXVY)
            U[:,4]=rever_transform(U[:,4],MINVX,MAXVX)
            U[:,5]=rever_transform(U[:,5],MINVY,MAXVY)              
            k2 = dt*U
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





    def running_cost(state,action, prev_state,dstate,reach_targets1,reach_targets2,reach_targets3,total_reward1,total_reward2,total_reward3,c1,c2,c3,i):
        position = state # K * nx
        # dstate K * nx; prev_state 1 * nx
        prev_position = prev_state.view(1,-1).repeat(position.shape[0],1)
        #print( prev_state)
        #print(c3)
        close1=c1.view(1,-1).repeat(position.shape[0],1)
        close2=c2.view(1,-1).repeat(position.shape[0],1)        
        close3=c3.view(1,-1).repeat(position.shape[0],1)


        id_non1 = np.nonzero(reach_targets1)
        id_non2 = np.nonzero(reach_targets2)
        id_non3 = np.nonzero(reach_targets3)
        if id_non1[0].size == 0:
            id_tar1 = 0;
        else:
            id_tar1 = id_non1[0][-1] + 1
           
        current_tar1 = POSITION_TAR1[id_tar1, :]
#       vec2 = current_tar - position
        if id_non2[0].size == 0:
            id_tar2 = 0;
        else:
            id_tar2 = id_non2[0][-1] + 1
           
        current_tar2 = POSITION_TAR2[id_tar2, :]        
#        print(reach_targets)
        if id_non3[0].size == 0:
            id_tar3 = 0;
        else:
            id_tar3 = id_non3[0][-1] + 1
           
        current_tar3 = POSITION_TAR3[id_tar3, :]         
        
        
        #vec1 = dstate-(torch.cat((close1,close2,close3),dim=1)-prev_position)
        #print(torch.cat((close1,close2,close3),dim=1)-prev_position)
        if id_tar1==0:
            posi_abs1=char_N1(state[:,:2])
            tang1=tang_N1(close1)
        elif id_tar1==1:
            posi_abs1=char_N2(state[:,:2])            
            tang1=tang_N2(close1)
        else:
            posi_abs1=char_N3(state[:,:2])             
            tang1=tang_N3(close1)
        
        if id_tar2==0:
            posi_abs2=char_U1(state[:,2:-2])            
            tang2=tang_U1(close2)
        elif id_tar2==1:
            posi_abs2=char_U2(state[:,2:-2])            
            tang2=tang_U2(close2)
        else:
            posi_abs2=char_U3(state[:,2:-2])             
            tang2=tang_U3(close2)
        
        if id_tar3==0:
            posi_abs3=char_S1(state[:,-2:])            
            tang3=tang_S1(close3)
        elif id_tar3==1:
            posi_abs3=char_S2(state[:,-2:])            
            tang3=tang_S2(close3)
        else:
            posi_abs3=char_S3(state[:,-2:])             
            tang3=tang_S3(close3)
        
        '''
        vec2 = torch.cat((tang1*pow((SN-total_reward1)/SN,powern),tang2*pow((SU-total_reward2)/SU,powern),tang3*pow((SS-total_reward3)/SS,powern)),dim=1)
        sum_vec2=pow((SN-total_reward1)/SN,powern)+pow((SU-total_reward2)/SU,powern)+pow((SS-total_reward3)/SS,powern)
        vec2=vec2/sum_vec2
        '''
        #vec2 = torch.cat((tang1*pow((SN-total_reward1)/SN,powern),tang2*pow((SU-total_reward2)/SU,powern),tang3*pow((SS-total_reward3)/SS,powern)),dim=1)
        
        #sum_vec2=pow((SN-total_reward1)/SN,powern)+pow((SU-total_reward2)/SU,powern)+pow((SS-total_reward3)/SS,powern)
        
        #vec2=vec2/sum_vec2
        #print(vec2)
        vec11= torch.sum(torch.mul(tang1, dstate[:,:2]), 1) / torch.norm(tang1, dim=1)
        vec12= torch.sum(torch.mul(tang2, dstate[:,2:4]), 1) / torch.norm(tang2, dim=1)
        vec13= torch.sum(torch.mul(tang3, dstate[:,4:]), 1) / torch.norm(tang3, dim=1)
        vec1=torch.cat((vec11.view(-1,1),vec12.view(-1,1),vec13.view(-1,1)),dim=1)
        vec2=torch.tensor([(SN-total_reward1)/SN,(SU-total_reward2)/SU,(SS-total_reward3)/SS])
        vec2=vec2.view(1,-1).repeat(position.shape[0],1)
        '''
        vec2 = torch.cat((tang1*math.exp((SN-total_reward1)/SN),tang2*math.exp((SU-total_reward2)/SU),tang3*math.exp((SS-total_reward3)/SS)),dim=1)
        '''
#        vec2 = current_tarPOSITION_INI - position
        costhe = torch.sum(torch.mul(vec1, vec2), 1) / (torch.norm(vec1, dim=1) * torch.norm(vec2, dim=1)) ##cannot neglect dim here
        #print(max(costhe))
        the = torch.acos(costhe)
#        cost_the = 1 - costhe ** 2 ## FAIL
        cost_the = 1 - costhe ## works well
        
        '''
        
        dist_circle = torch.sum(position ** 2, dim=1) - 0.25
#        print("dist")
#       print(dist_circle)
        dist_circle[dist_circle < 0] = 0
        '''
#        prev_dist_sqr = torch.sum ((prev_position - current_tar)**2  , dim=1)
        #cost = WEIGHT_ANGLE * cost_the
        
#        cost = WEIGHT_POSITION * torch.sum(((position - current_tar) ** 2) , dim=1)  + WEIGHT_ACTION * torch.sum(action ** 2, dim=1)  + WEIGHT_ANGLE * cost_the
#        cost1=torch.sum(((position[:,:2] - position[:,2:])**2 ) , dim=1)
#        cost2=torch.sum(((prev_position[:,:2] - prev_position[:,2:])**2 ) , dim=1)
        
        cost = WEIGHT_POSITION *(posi_abs1+posi_abs2+posi_abs3)+ WEIGHT_ACTION * torch.sum(action ** 2, dim=1) + WEIGHT_ANGLE * cost_the
#        cost = WEIGHT_POSITION * torch.sum(((position - current_tar) ** 2) , dim=1) + WEIGHT_ACTION * torch.sum(action ** 2, dim=1) + WEIGHT_ANGLE * the ** 2        
#        cost = WEIGHT_POSITION * torch.sum((position - POSITION_TAR) ** 2, dim=1) + WEIGHT_ACTION * torch.sum(action ** 2, dim=1) +  C_OUTBOUND * dist_circle
        costlist=torch.cat((prev_position,state,cost.view(-1,1)),dim=1)
        torch.save(costlist, 'cost'+str(i)+'.pt')

        return cost            
            

        
    def retrain(new_data):
        pass

    

    

        


    efficiency1=0
    efficiency2=0    
    efficiency3=0     
#    env.state = np.array([X_ini, Y_ini,X_ini2, Y_ini2])
    for e in range(1):
#    for e in range(N_EPISODES):
#        print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
#        print(e)

        
        
        
        
        
       
        X_ini1=-(L+3*R)
        Y_ini1=-R
        X_ini2=-R
        Y_ini2=R
        X_ini3=3*R+L
        Y_ini3=0.5*R      

        
        reward1=SN
        reward2=SU
        reward3=SS
        '''
        X_tar1 = [-(R/2.0+2*R),-(R/2.0+R),-(R/2.0+R)]
        Y_tar1 = [0.0+1.5*R,-R+1.5*R,0.0+1.5*R]
        X_tar2 = [-(RU/2.0),(RU/2.0),(RU/2.0)]
        Y_tar2 = [-(RU/2.0)*math.sqrt(3)+1.5*R,-(RU/2.0)*math.sqrt(3)+1.5*R,0.0+1.5*R]
        X_tar3 = [1.5*R,2.5*R,1.5*R]
        Y_tar3 = [(0.5*math.sqrt(3)-1.0)*R+1.5*R,-0.5*math.sqrt(3)*R+1.5*R,-0.5*math.sqrt(3)*R+1.5*R]
        '''
        X_tar1 = [-(L+3*R),-(L+R),-(L+R)]
        Y_tar1 = [R,-R,R]
        X_tar2 = [-R, R,R]
        Y_tar2 = [0,0,R]
        X_tar3 = [L+R,L+2*R,L+R]
        Y_tar3 = [R*0.5,0,-R*0.5]        
                    
        N_tar1=len(X_tar1)
        N_tar2=len(X_tar2)
        N_tar3=len(X_tar3)
        POSITION_INI = torch.Tensor([[X_ini1], [Y_ini1],[X_ini2], [Y_ini2],[X_ini3], [Y_ini3]]).t()
        POSITION_TAR1 = torch.Tensor([X_tar1, Y_tar1]).t()
        POSITION_TAR2 = torch.Tensor([X_tar2, Y_tar2]).t()
        POSITION_TAR3 = torch.Tensor([X_tar3, Y_tar3]).t()         
        env = gym.make(ENV_NAME)  # bypass the default TimeLimit wrapper

        '''
        if reward<0.1:
            continue
        '''
        
        env.reset()
        env.state = np.array([X_ini1, Y_ini1,X_ini2, Y_ini2,X_ini3, Y_ini3])
        ##overwrite environment variables.
        env.dt = DT
        env.dtt=dt
        env.n = N_INLETS
        env.X_ini1 = X_ini1
        env.Y_ini1 = Y_ini1
        env.X_tar1 = X_tar1
        env.Y_tar1 = Y_tar1
        env.X_ini2 = X_ini2
        env.Y_ini2 = Y_ini2
        env.X_tar2 = X_tar2
        env.Y_tar2 = Y_tar2
        env.X_ini3 = X_ini3
        env.Y_ini3 = Y_ini3
        env.X_tar3 = X_tar3
        env.Y_tar3 = Y_tar3        
        env.x_tar1 = np.array([X_tar1, Y_tar1], dtype=np.float32)
        env.x_tar2 = np.array([X_tar2, Y_tar2], dtype=np.float32)
        env.x_tar3 = np.array([X_tar3, Y_tar3], dtype=np.float32)        ## 2 rows * ntargets columns
        env.dist_epsi_goal = DIST_EPSI_GOAL
        env.ACTION_LOW = ACTION_LOW
        env.ACTION_HIGH = ACTION_HIGH
        env.MINVX=MINVX
        env.MINVY=MINVY
        env.MAXVX=MAXVX
        env.MAXVY=MAXVY        
        env.n=N_INLETS
        env.closest1=np.array([X_ini1, Y_ini1], dtype=np.float32)
        env.closest2=np.array([X_ini2, Y_ini2], dtype=np.float32)        
        env.closest3=np.array([X_ini3, Y_ini3], dtype=np.float32)        
        
        
        
        
        env.state = np.array([X_ini1, Y_ini1,X_ini2, Y_ini2,X_ini3, Y_ini3])      
        if rank==root_rank:
            mppi_gym = nppi_trap_ntar3t_NN.MPPI(dynamics, true_dynamics, running_cost,nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                                 lambda_=lambda_, device=d, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
                                 u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))
        else:
            mppi_gym = None
        mppi_gym = comm.bcast(mppi_gym, root=root_rank) 
        total_reward1,total_reward2, total_reward3,_, traj, done, iter_act, r_targets1, r_targets2,r_targets3= nppi_trap_ntar3t_NN.run_mppi(mppi_gym, env,rank,root_rank,comm,retrain ,ifzero = ZERO_AFTER_RETRAIN, iter = N_ITERATIONS)
        if rank==root_rank: 
            n_targets_reach1 = np.nonzero(r_targets1)[0].size
            n_targets_reach2 = np.nonzero(r_targets2)[0].size
            n_targets_reach3 = np.nonzero(r_targets3)[0].size        
            if n_targets_reach1 == N_tar1 or n_targets_reach2 == N_tar2 or n_targets_reach3 == N_tar3:
                succeed = True
                efficiency1=total_reward1/reward1
                efficiency2=total_reward2/reward2          
                efficiency3=total_reward3/reward3             
            else:
                succeed = False
                 
            



            print(e, total_reward1,total_reward2,total_reward3,done, succeed, n_targets_reach1, n_targets_reach2,n_targets_reach3,iter_act)
            traj_bin = os.path.join(traj_path,"trajectory_"+str(e)+".bin")
            n_record_per_traj = os.path.join(traj_path,"n_records_"+str(e)+".bin")
            ini_tar_bin = os.path.join(traj_path,"ini_tar_"+str(e)+".bin")
            np.array(traj.shape[1]).astype('int').tofile(n_record_per_traj)        
            traj.astype('float').tofile(traj_bin)
            
            ini_tar = np.array(([N_tar1] + [X_ini1] + X_tar1 + [Y_ini1] + Y_tar1+[X_ini2] + X_tar2 + [Y_ini2] + Y_tar2+[X_ini3] + X_tar3 + [Y_ini3] + Y_tar3))
            ini_tar.astype('float').tofile(ini_tar_bin)                
#        nE = range(len(Error))
#        plt.plot(Error)
#        plt.show()


            print(efficiency1,efficiency2,efficiency3)
