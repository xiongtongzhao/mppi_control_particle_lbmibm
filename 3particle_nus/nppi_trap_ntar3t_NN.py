import torch
import time
import logging
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import mpi4py as MPI
import gym
import gym_particle



def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))


class MPPI():
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(self, dynamics, true_dynamics, running_cost,nx, noise_sigma, num_samples=100, horizon=15, device="cpu",
                 terminal_state_cost=None,
                 lambda_=1.,
                 noise_mu=None,
                 u_min=None,
                 u_max=None,
                 u_init=None,
                 U_init=None,
                 u_scale=1,
                 u_per_command=1,
                 step_dependent_dynamics=False,
                 dynamics_variance=None,
                 running_cost_variance=None,
                 sample_null_action=False):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K x 1) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        :param lambda_: temperature, positive scalar where larger values will allow more exploration
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param step_dependent_dynamics: whether the passed in dynamics needs horizon step passed in (as 3rd arg)
        :param dynamics_variance: function(state) -> variance (K x nx) give variance of the state calcualted from dynamics
        :param running_cost_variance: function(variance) -> cost (K x 1) cost function on the state variances
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        """
        self.d = device
        self.dtype = noise_sigma.dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS
        self.nps=2
        # dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) is 0 else noise_sigma.shape[0]
        self.lambda_ = lambda_

        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)

        if u_init is None:
            u_init = torch.zeros_like(noise_mu)

        # handle 1D edge case
        if self.nu is 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        # bounds
        self.u_min = u_min
        self.u_max = u_max
        self.u_scale = u_scale
        self.u_per_command = u_per_command
        # make sure if any of them is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            if not torch.is_tensor(self.u_max):
                self.u_max = torch.tensor(self.u_max)
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            if not torch.is_tensor(self.u_min):
                self.u_min = torch.tensor(self.u_min)
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)

        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
        #self.noise_mu, nu
        
        # T x nu control sequence
        self.U = U_init
        self.u_init = u_init.to(self.d)

        if self.U is None:
            self.U = self.noise_dist.sample((self.T,)) # self.U: T * nu

        self.step_dependency = step_dependent_dynamics
        self.F = dynamics
        self.F_true = true_dynamics # added by lailai
        self.dynamics_variance = dynamics_variance
        self.running_cost = running_cost

        self.running_cost_variance = running_cost_variance
        self.terminal_state_cost = terminal_state_cost
        self.sample_null_action = sample_null_action
        self.state = None

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None
        if self.dynamics_variance is not None and self.running_cost_variance is None:
            raise RuntimeError("Need to give running cost for variance when giving the dynamics variance")

    def _dynamics(self, state, u, t):
#        return self.F_true(state, u, t) if self.step_dependency else self.F_true(state, u) # added by lailai
        return self.F(state, u, t) if self.step_dependency else self.F(state, u) # original
    def command(self, state_targets,reward1,reward2,reward3,c1,c2,c3,i):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :returns action: (nu) best action
        """
        state = state_targets[0]
        self.reach_targets1 = state_targets[1]
        self.reach_targets2 = state_targets[2]
        self.reach_targets3 = state_targets[3]
        # shift command 1 time step
        
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init
        
        if not torch.is_tensor(state):
            #print(state)
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)

        cost_total = self._compute_total_cost_batch(reward1,reward2,reward3,c1,c2,c3,i)

#        print(cost_total)
        beta = torch.min(cost_total)
        self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)
#        print(self.cost_total_non_zero)
        
        eta = torch.sum(self.cost_total_non_zero)
#        print([eta, self.cost_total_non_zero.nonzero().shape])
        self.omega = (1. / eta) * self.cost_total_non_zero
        for t in range(self.T):
            self.U[t] += torch.sum(self.omega.view(-1, 1) * self.noise[:, t], dim=0)
        action = self.U[:self.u_per_command]

        return action

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = self.noise_dist.sample((self.T,)) # self.U: T * nu

    def _compute_total_cost_batch(self,reward1,reward2,reward3,c1,c2,c3,i):
        if not torch.is_tensor(c1):
            c1 = torch.tensor(c1)
        if not torch.is_tensor(c2):
            c2 = torch.tensor(c2)            
        if not torch.is_tensor(c3):
            c3 = torch.tensor(c3)            
        # parallelize sampling across trajectories
        self.cost_total = torch.zeros(self.K, device=self.d, dtype=self.dtype)

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (self.K, self.nx):
            state = self.state
        else:## a single state
            state = self.state.view(1, -1).repeat(self.K, 1) ## shape to K * nx, every row is identical to each other.

        # resample noise each time we take an action
        self.noise = self.noise_dist.sample((self.K, self.T)) # K * T * nu
        # broadcast own control to noise over samples; now it's K x T x nu
        self.perturbed_action = self.U + self.noise #U: T * nu; noise: K * T * nu
        if self.sample_null_action:
            self.perturbed_action[self.K - 1] = 0
        # naively bound control
        self.perturbed_action = self._bound_action(self.perturbed_action)
        torch.save(self.perturbed_action, 'action'+str(i)+'.pt')
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U
        action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv

        self.states = []
        self.actions = []
        for t in range(self.T):
            u = self.u_scale * self.perturbed_action[:, t] ## K * nu
            ##state : K * nx
            [state, d_state] = self._dynamics(state, u, t)
## d_state added by lailai
            #state: torch, K * nx
            #u: torch, K * nu




            if t == 0:
                prev_state = (state - d_state)[0,:].unsqueeze(0)
#                print(prev_state.shape)
#            self.cost_total += self.running_cost3(state[:,:self.nps],state[:,self.nps:], self.reach_targets)
            '''
            self.cost_total += self.running_cost3(state[:,:self.nps],state[:,self.nps:],prev_state[:,:self.nps],prev_state[:,self.nps:], self.reach_targets)
            self.cost_total += self.running_cost1(state[:,:self.nps], u, d_state[:,:self.nps], prev_state[:,:self.nps], self.reach_targets)
            self.cost_total += self.running_cost2(state[:,self.nps:], u, d_state[:,self.nps:], prev_state[:,self.nps:], self.reach_targets)
            '''
            self.cost_total +=self.running_cost(state, u, prev_state,d_state,self.reach_targets1 ,self.reach_targets2,self.reach_targets3,reward1,reward2,reward3,c1,c2,c3,i)
            ## d_state added by lailai
            if self.dynamics_variance is not None:
                self.cost_total += self.running_cost_variance(self.dynamics_variance(state))

            # Save total states/actions
            self.states.append(state)
            self.actions.append(u)
            ##self.states: a list of length T, each element is torch of size  K * nx
            ##self.actions: a list of length T, each element is torch of size  K * nu
            
        #print(self.cost_total)
        # Actions is K x T x nu
        # States is K x T x nx
        self.actions = torch.stack(self.actions, dim=1) ##stack: concatenate the list of T tensors in the dim-th dimention.
        self.states = torch.stack(self.states, dim=1)

        # action perturbation cost
        if self.terminal_state_cost:
            terminal_cost, self.actions = self.terminal_state_cost(self.states, self.actions)
            self.cost_total += terminal_cost
        self.actions /= self.u_scale
        perturbation_cost = torch.sum(self.actions * action_cost, dim=(1, 2)) ## this might be negative values
        #print([torch.max(self.cost_total),torch.max(perturbation_cost)])
        self.cost_total += perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.u_max is not None:
            for t in range(self.T):
                u = action[:, self._slice_control(t)]
                cu = torch.max(torch.min(u, self.u_max), self.u_min)
                action[:, self._slice_control(t)] = cu
        return action

    def _slice_control(self, t):
        return slice(t * self.nu, (t + 1) * self.nu)

    def get_rollouts(self, state, num_rollouts=1):
        """
            :param state: either (nx) vector or (num_rollouts x nx) for sampled initial states
            :param num_rollouts: Number of rollouts with same action sequence - for generating samples with stochastic
                                 dynamics
            :returns states: num_rollouts x T x nx vector of trajectories

        """
        state = state.view(-1, self.nx)
        if state.size(0) == 1:
            state = state.repeat(num_rollouts, 1)

        T = self.U.shape[0]
        states = torch.zeros((num_rollouts, T + 1, self.nx), dtype=self.U.dtype, device=self.U.device)
        states[:, 0] = state
        for t in range(T):
            states[:, t + 1] = self._dynamics(states[:, t].view(num_rollouts, -1),
                                              self.u_scale * self.U[t].view(num_rollouts, -1), t)[0] ## [0] added by lailai to get the state, [1] are d_state
        return states[:, 1:]


def run_mppi(mppi, ENV,rank,rrank,comm,retrain_dynamics, retrain_after_iter=50, iter=1000, ifzero=False, render=True):
    #run one episode
    if rank==rrank:
        env = ENV
 
    else:
        env=None    
    if rank ==rrank:    
        dataset = torch.zeros((retrain_after_iter, mppi.nx + mppi.nu), dtype=mppi.U.dtype, device=mppi.d)
        total_reward1 = 0
        total_reward2 = 0
        total_reward3 = 0    
        trajectory = []
        done = False
        reach_targets1 = None
        reach_targets2 = None
        reach_targets3 = None
        c1=np.array([env.X_ini1,env.Y_ini1])
        c2=np.array([env.X_ini2,env.Y_ini2])
        c3=np.array([env.X_ini3,env.Y_ini3])
    for i in range(iter):
#        print(i)
#        print(i)
        if rank ==rrank: 
            if done:
                break
            state = env.state.copy()
            command_start = time.perf_counter()       
            
            if reach_targets1 is None:
                reach_targets1 = np.zeros(env.ntargets1)
            if reach_targets2 is None:
                reach_targets2 = np.zeros(env.ntargets2)        
            if reach_targets3 is None:
                reach_targets3 = np.zeros(env.ntargets3)         
           # print(reach_targets)
            
            action = mppi.command([state, reach_targets1,reach_targets2,reach_targets3],total_reward1,total_reward2,total_reward3,c1,c2,c3,i)
            elapsed = time.perf_counter() - command_start
        else:
            action=None
        ###collecting the data from true dynamics

#        print(action.detach().numpy())
        env= comm.bcast(env, root = rrank)
        action=comm.bcast(action,root=rrank)        
        obs, r1, r2,r3, c1,c2,c3,done, _ = env.step(action.detach().numpy())
        #obs: np array
        if rank ==rrank:         
            reach_targets1 = obs[mppi.nx:mppi.nx+env.ntargets1] #n_tars
            reach_targets2 = obs[mppi.nx+env.ntargets1:mppi.nx+env.ntargets1+env.ntargets2]
            reach_targets3 = obs[mppi.nx+env.ntargets1+env.ntargets2:]      
            
            result_step = np.concatenate((obs[:mppi.nx], action.detach().numpy(), np.array(r1+r2+r3), int(done)), axis = None)
            #print(s)
            if ( i==0 ):
                trajectory = result_step
            else:
                trajectory = np.vstack( (trajectory, result_step) )

            total_reward1 = r1
            total_reward2 = r2
            total_reward3 = r3        

            if render:
                env.render()
        if rank ==rrank: 
            ##the dataset is cleaned and collect data from true dynamics every 'retrain_after_iter' steps.
            di = i % retrain_after_iter
            if di == 0 and i > 0:
                retrain_dynamics(dataset)
                # don't have to clear dataset since it'll be overridden, but useful for debugging
                if ifzero:
                    dataset.zero_()
            
            #print(dataset[di, :mppi.nx].shape)
            #print(torch.tensor(state, dtype=mppi.U.dtype).shape)
            
            dataset[di, :mppi.nx] = torch.tensor(state, dtype=mppi.U.dtype).reshape(dataset[di, :mppi.nx].shape)            
            dataset[di, mppi.nx:] = action
        
    #env.close()
    #mppi.reset()
    if rank ==rrank:
        niter_actual = i + 1
    if rank ==rrank:        
        return total_reward1,total_reward2, total_reward3, dataset, trajectory, done, niter_actual, reach_targets1,reach_targets2,reach_targets3
    else:
        return None,None,None,None,None,None,None,None,None,None       
