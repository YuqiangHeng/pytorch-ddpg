
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (SubActionActor, SubActionActorParallel, SubActionCritic, SubActionCriticParallel)
from memory import SequentialMemory,BeamSpaceSubActionSequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# from ipdb import set_trace as debug

criterion = nn.MSELoss()


class SubAction_DDPG(object):
    def __init__(self, nb_states, nb_actions, args):
        self.seed = args.seed
        if self.seed > 0:
            self.seed(self.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        self.num_beams_per_UE = args.num_beams_per_UE
        self.num_measurements = args.num_measurements
        self.window_length = args.window_length
#        self.combine_state = args.combine_state        
        # Create Actor and Critic Network
        # self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        # self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        # self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        # self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        # self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        # self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)
        
        self.actor = SubActionActorParallel(self.nb_states, self.nb_actions, self.window_length, self.num_measurements).to(device)
        self.actor_target = SubActionActorParallel(self.nb_states, self.nb_actions, self.window_length, self.num_measurements).to(device)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)
        
        self.critic = SubActionCriticParallel(self.nb_states, self.nb_actions, self.window_length, self.num_measurements).to(device)
        self.critic_target = SubActionCriticParallel(self.nb_states, self.nb_actions, self.window_length, self.num_measurements).to(device)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        # self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.memory = BeamSpaceSubActionSequentialMemory(limit=args.rmsize, window_length=self.window_length, num_measurements=self.num_measurements)
        self.random_process = OrnsteinUhlenbeckProcess(size=self.actor.outshape, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

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
        self.debug = args.debug
        self.actor_lambda = args.actor_lambda
        self.policy_update_counter = 0
        self.training_log = {'critic_mse':[],'actor_mse':[],'actor_value':[],'actor_total':[]}

        # 
        # if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, subaction_rewards_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        with torch.no_grad():
            next_states = torch.from_numpy(next_state_batch).type(torch.FloatTensor).to(device)
            target_actor_output = self.actor_target(next_states) 
            # next_actions = torch.from_numpy(self.pick_beams_batch(to_numpy(predicted_beam_qual_target)))
            next_actions = self.actor_target.select_beams(to_numpy(target_actor_output), self.num_beams_per_UE)
            next_subq_values, next_q_values = self.critic_target([next_states,next_actions])
        # next_q_values.volatile=False
        next_subq_values.requires_grad = True
        next_q_values.requires_grad = True
        
        target_subq_batch = torch.from_numpy(subaction_rewards_batch).to(device) + \
            self.discount*torch.from_numpy(terminal_batch.astype(np.float)).to(device)*next_q_values.expand(-1, self.nb_actions)/self.num_beams_per_UE
            
        target_q_batch = torch.from_numpy(reward_batch).to(device) + \
            self.discount*torch.from_numpy(terminal_batch.astype(np.float)).to(device)*next_q_values

        # Critic update
        self.critic.zero_grad()
        subq_batch, q_batch = self.critic([torch.from_numpy(state_batch).type(torch.FloatTensor).to(device), action_batch])
        subvalue_loss = criterion(subq_batch, target_subq_batch)
        # subvalue_loss = torch.tensor(0)
        value_loss = criterion(q_batch, target_q_batch)
        total_loss = 0.5*subvalue_loss + 0.5*value_loss
        # total_loss = value_loss
        # print(value_loss.item())
        total_loss.backward()
        self.training_log['critic_mse'].append([value_loss.item(),subvalue_loss.item()])
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()
        
        # Beam qual prediction loss, only if using MSE of actor output
        states = torch.from_numpy(state_batch).type(torch.FloatTensor).to(device)
        actor_output = self.actor(states)
        # predicted_beam_qual = self.actor(states)
        # broadcasted_action_batch = np.expand_dims(action_batch, axis=1)
        # broadcasted_action_batch = np.repeat(broadcasted_action_batch, self.num_measurements, axis=1)
        # actor_output_masked = torch.mul(actor_output, to_tensor(broadcasted_action_batch))
        # true_beam_qual = to_tensor(next_state_batch[:,-self.num_measurements:,:])
        # actor_output_loss = criterion(actor_output_masked,true_beam_qual)        
        # self.training_log['actor_mse'].append(actor_output_loss.item())
        
        # Value loss
        # actions = torch.from_numpy(self.pick_beams_batch(to_numpy(predicted_beam_qual)))
        actions = self.actor.select_beams(to_numpy(actor_output),self.num_beams_per_UE)
        subaction_policy_loss, policy_loss = self.critic([states,actions])
        policy_loss = -policy_loss.mean()
        self.training_log['actor_value'].append(policy_loss.item())
        policy_loss.backward()
        
        
        # # Combine actor output MSE and -value from critic
        # total_loss = (1-self.actor_lambda)*policy_loss + self.actor_lambda * actor_output_loss
        # self.training_log['actor_total'].append(total_loss.item())
        # total_loss.backward()
        
        #take a gradient step
        self.actor_optim.step()
        # if self.debug and self.policy_update_counter % 20 == 0:
        #     print('Critic Loss: {}. Actor Prediction Loss: {}. Actor Policy Loss: {}'.format(value_loss.item(), beam_qual_prediction_loss.item(),policy_loss.item()))
        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        self.policy_update_counter += 1

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

    def observe(self, subaction_r_t, r_t, ob_t1, done):
#        if self.is_training:
#            if self.combine_state:
#                combined_s_t = np.concatenate((self.ob_t, self.a_t), axis=0)
#                print(self.a_t)
#                self.memory.append(combined_s_t, self.a_t, r_t, done)
#            else:
#                self.memory.append(self.ob_t, self.a_t, r_t, done)
#            self.ob_t = ob_t1
        if self.is_training:
            self.memory.append(self.ob_t, self.a_t, subaction_r_t, r_t, done)
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
        with torch.no_grad():
    #        s_t = self.memory.get_recent_state(np.concatenate((observation, self.a_t),axis=0))
            s_t = self.memory.get_recent_state(observation)
            #remove existing empty dimension and add batch dimension 
            s_t_array = np.array([np.squeeze(np.array(s_t))])
            s_t_array_tensor = torch.from_numpy(s_t_array).type(torch.FloatTensor).to(device)
            actor_output = to_numpy(self.actor(s_t_array_tensor)) #bsize(1) x actor_output_shape
            actor_output += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
            action = self.actor.select_beams(actor_output, self.num_beams_per_UE).squeeze(0)
            # action = to_numpy(
            #     self.actor(to_tensor(np.array([s_t])))
            # ).squeeze(0)
            # action = np.clip(action, -1., 1.)
    
            if decay_epsilon:
                self.epsilon -= self.depsilon
                
            self.a_t = action
        return action
    
    # def pick_beams(self, observation:np.ndarray):
    #     #observation is batchsize x num_measurements x num_beams matrix, iteratively pick best beam
    #     selected_beams = np.argsort(np.sum(observation,axis=0))[-self.num_beams_per_UE:]
    #     binary_beams = np.zeros(self.nb_actions)
    #     binary_beams[selected_beams] = 1
        
    #     # selected_beams = []
    #     # pool = list(np.arange(self.nb_actions))
    #     # sum_tp = np.sum(observation,axis=0)
    #     # sel_beam = np.argmax(sum_tp)
    #     # selected_beams.append(sel_beam)
    #     # pool.remove(sel_beam)
        
    #     # for it_idx in range(self.num_beams_per_UE):
    #     #     sum_tp = np.sum(observation[pool,:],axis=0)
    #     #     sel_beam = np.argmax(sum_tp)
    #     #     selected_beams.append(sel_beam)
    #     #     pool.remove(sel_beam)
            
    #     return binary_beams
    
    # def pick_beams_batch(self, observation:np.ndarray):
    #     assert(observation.shape[0] == self.batch_size)
    #     binary_beams = np.zeros((self.batch_size, self.nb_actions))
    #     for i in range(self.batch_size):
    #         binary_beams[i,:] = self.pick_beams(observation[i])
    #     return binary_beams
    # def reset(self, obs):
    #     self.ob_t = obs
    #     self.random_process.reset_states()
     
    # modified reset() function that also takes in initial beam config
    def reset(self, obs):
        self.ob_t = obs
        # self.a_t = beams
        self.random_process.reset_states()
        # self.policy_update_counter = 0
        # self.training_log = {'critic_mse':[],'actor_mse':[],'actor_value':[],'actor_total':[]}

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
