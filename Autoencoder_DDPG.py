
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic,MLP,SerializedCritic)
from memory import SequentialMemory,BeamSpaceSequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *

# from ipdb import set_trace as debug

criterion = nn.MSELoss()


class Autoencoder_DDPG(object):
    def __init__(self, nb_states, nb_actions, args):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        self.num_beams_per_UE = args.num_beams_per_UE
#        self.combine_state = args.combine_state        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w,
            'window_len':args.window_length * args.num_measurements,
            'kernel_size_1':args.conv2d_1_kernel_size,
            'kernel_size_2':args.conv2d_2_kernel_size
        }
        # self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        # self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        # self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        # self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        # self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        # self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)
        
        self.actor = MLP(self.nb_states, args.window_length, args.num_measurements)
        self.actor_target = MLP(self.nb_states, args.window_length, args.num_measurements)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)
        self.critic = SerializedCritic(self.nb_states, self.nb_actions, args.window_length, args.num_measurements)
        self.critic_target = SerializedCritic(self.nb_states, self.nb_actions, args.window_length, args.num_measurements)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        # self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.memory = BeamSpaceSequentialMemory(limit=args.rmsize, window_length=args.window_length, num_measurements=args.num_measurements)
        self.random_process = OrnsteinUhlenbeckProcess(size=(args.num_measurements,nb_states), theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.ob_t = None # Most recent observation
        self.a_t = None # Most recent action
        self.is_training = True

        # 
        if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_states = to_tensor(next_state_batch, volatile=True)
        next_actor_outputs = self.actor_target(to_tensor(next_state_batch, volatile=True))
        next_actions = to_tensor(self.pick_beams_batch(to_numpy(next_actor_outputs)))
        next_q_values = self.critic_target([next_states,next_actions])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        # print(value_loss.item())
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        states = to_tensor(state_batch)
        actor_outputs = self.actor(states)
        actions = to_tensor(self.pick_beams_batch(to_numpy(actor_outputs)))
        policy_loss = -self.critic([states,actions])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, ob_t1, done):
#        if self.is_training:
#            if self.combine_state:
#                combined_s_t = np.concatenate((self.ob_t, self.a_t), axis=0)
#                print(self.a_t)
#                self.memory.append(combined_s_t, self.a_t, r_t, done)
#            else:
#                self.memory.append(self.ob_t, self.a_t, r_t, done)
#            self.ob_t = ob_t1
        if self.is_training:
            self.memory.append(self.ob_t, self.a_t, r_t, done)
            self.ob_t = ob_t1
    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)
        binary_action = np.zeros(self.nb_actions)
        binary_action[np.argsort(action)[-self.num_beams_per_UE:]]=1
        self.a_t = binary_action
        return binary_action

    # def select_action(self, s_t, decay_epsilon=True):
    #     action = to_numpy(
    #         self.actor(to_tensor(np.array([s_t])))
    #     ).squeeze(0)
    #     action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
    #     action = np.clip(action, -1., 1.)

    #     if decay_epsilon:
    #         self.epsilon -= self.depsilon
        
    #     self.a_t = action
    #     return action
 
    # a modified implementation of selection_action that enables window_length > 1
    def select_action(self, observation, decay_epsilon=True):
#        s_t = self.memory.get_recent_state(np.concatenate((observation, self.a_t),axis=0))
        s_t = self.memory.get_recent_state(observation)
        #remove existing empty dimension and add batch dimension 
        s_t_array = np.array([np.squeeze(np.array(s_t))])
        # action = to_numpy(
        #     self.actor(to_tensor(s_t_array))
        # ).squeeze(0)
        actor_output = to_numpy(self.actor(to_tensor(s_t_array))).squeeze(0)
        actor_output += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action = self.pick_beams(actor_output)     
        # action = to_numpy(
        #     self.actor(to_tensor(np.array([s_t])))
        # ).squeeze(0)
        # action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
            
        self.a_t = action
        return action
    
    def pick_beams(self, observation:np.ndarray):
        #observation is batchsize x num_measurements x num_beams matrix, iteratively pick best beam
        selected_beams = np.argsort(np.sum(observation,axis=0))[-self.num_beams_per_UE:]
        binary_beams = np.zeros(self.nb_actions)
        binary_beams[selected_beams] = 1
        
        # selected_beams = []
        # pool = list(np.arange(self.nb_actions))
        # sum_tp = np.sum(observation,axis=0)
        # sel_beam = np.argmax(sum_tp)
        # selected_beams.append(sel_beam)
        # pool.remove(sel_beam)
        
        # for it_idx in range(self.num_beams_per_UE):
        #     sum_tp = np.sum(observation[pool,:],axis=0)
        #     sel_beam = np.argmax(sum_tp)
        #     selected_beams.append(sel_beam)
        #     pool.remove(sel_beam)
            
        return binary_beams
    
    def pick_beams_batch(self, observation:np.ndarray):
        assert(observation.shape[0] == self.batch_size)
        binary_beams = np.zeros((self.batch_size, self.nb_actions))
        for i in range(self.batch_size):
            binary_beams[i,:] = self.pick_beams(observation[i])
        return binary_beams
    # def reset(self, obs):
    #     self.ob_t = obs
    #     self.random_process.reset_states()
     
    # modified reset() function that also takes in initial beam config
    def reset(self, obs):
        self.ob_t = obs
        # self.a_t = beams
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
