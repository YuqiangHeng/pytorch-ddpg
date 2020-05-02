# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:35:06 2020

@author: ethan
"""

import argparse
import gym
import numpy as np
from itertools import count
from memory import BeamSpaceSequentialMemory

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Multinomial
from BeamManagementEnv import BeamManagementEnv
import time
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt

# USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

eps = np.finfo(np.float32).eps.item()

class Policy(nn.Module):
    def __init__(self, nb_states, window_len, num_measurements):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_shape = (window_len*num_measurements, nb_states)
        nsize = self._serial_input_size(self.input_shape)
        hidden1 = int(nsize/4)
        hidden2 = int(nsize/16)
        hidden3 = int(nsize/64)
        hidden4 = int(nsize/48)
        # hidden5 = int(nsize/32)
        self.outdim = nb_states
        self.outshape = nb_states
        self.saved_log_probs = []
        self.rewards = []
        self.fc1 = nn.Sequential(nn.Linear(nsize,hidden1),
                                    nn.Sigmoid())
        self.fc2 = nn.Sequential(nn.Linear(hidden1,hidden2),
                                    nn.Sigmoid())
        self.fc3 = nn.Sequential(nn.Linear(hidden2,hidden3),
                                    nn.Sigmoid())
        # self.fc4 = nn.Sequential(nn.Linear(hidden3,hidden4),
        #                             nn.ReLU())
        self.fc5 = nn.Sequential(nn.Linear(hidden3,hidden4),
                                    nn.Sigmoid())
        self.fc6 = nn.Sequential(nn.Linear(hidden4,self.outdim),
                                 nn.Sigmoid())

    def _serial_input_size(self, shape):
        batch_size = 1
        dummy_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        dummy_flat = self.flatten(dummy_input)
        nsize = dummy_flat.data.view(batch_size,-1).size(1)
        return nsize
    
    # def select_beams(self, x, nb_actions, n):
    #     bsize = x.shape[0]
    #     binary_beams = np.zeros((bsize, nb_actions))
    #     for i in range(bsize):
    #         sel = np.argsort(x[i])[-n:]
    #         binary_beams[i,sel] = 1
    #     return binary_beams
    
    def forward(self,x):
        flat = self.flatten(x)
        out = self.fc1(flat)
        out = self.fc2(out)
        out = self.fc3(out)
        # out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        return F.softmax(out)

class REINFORCE_Agent(object):
    def __init__(self,args):
        self.nb_states = args.num_antennas*args.oversampling_factor
        self.nb_actions = args.num_antennas*args.oversampling_factor
        self.window_length = args.window_length
        self.num_measurements = args.num_measurements
        self.num_beams_per_UE = args.num_beams_per_UE
        self.policy = Policy(self.nb_states,self.window_length,self.num_measurements)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
                
    def eval_select_action(self,state:np.ndarray):
        with torch.no_grad():
            probs = self.policy(to_tensor(state).unsqueeze(0)).squeeze()
            action = np.argsort(to_numpy(probs))[-self.num_beams_per_UE:]
            binary_action = np.zeros(self.nb_actions)
            binary_action[action] = 1            
        return binary_action
            
        
    def sample_topk(self,probs):
        
        # #sample K times until get k distinct items then deduplicate
        # m = Categorical(probs)
        # sel = []
        # K = 0
        # while len(np.unique(sel)) < k:
        #     b = m.sample()
        #     sel.append(b.item())
        #     K += 1
        # action = np.unique(sel)
        # unselected = [i for i in range(len(probs)) if not i in action]
        # # p_action = 0
        # # for i in unselected:
        # #     p_action += (1-probs[i].item())**K
        # p_action = (1-probs[unselected].sum())**K
        
        #sample k times: remove sampled item and normalize probs
        p_action = 1
        sel = []
        pool = list(np.arange(len(probs)))
        probs_pool = probs[np.array(pool)]
        for i in range(self.num_beams_per_UE):
            m = Categorical(probs_pool)
            try:
                b = m.sample()
            except:
                print(probs_pool)
            sel.append(pool[b.item()])
            pool.remove(pool[b.item()])
            p_action *= probs_pool[b.item()]
            probs_pool = probs[np.array(pool)]/probs[np.array(pool)].sum()    
        action = sel
       
        log_p_action = torch.log(p_action)
        binary_action = np.zeros(self.nb_actions)
        binary_action[action] = 1
        return binary_action, log_p_action
    
    def select_action(self,state:np.ndarray):
        state = to_tensor(state).unsqueeze(0)
        probs = self.policy(state).squeeze()
        # tic = time.time()
        action, log_p_action = self.sample_topk(probs)
        # toc = time.time()
        # print('Time taken to sample top-K = {} seconds'.format(toc-tic))
        # m = Categorical(probs)
        # action = m.sample()
        # policy.saved_log_probs.append(m.log_prob(action))
        self.policy.saved_log_probs.append(log_p_action)
        return action

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = to_tensor(np.array(returns))
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        # print('Policy Loss = {}'.format(policy_loss.item()))
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]
        
# class Policy(nn.Module):
#     def __init__(self):
#         super(Policy, self).__init__()
#         self.affine1 = nn.Linear(4, 128)
#         self.dropout = nn.Dropout(p=0.6)
#         self.affine2 = nn.Linear(128, 2)

#         self.saved_log_probs = []
#         self.rewards = []

#     def forward(self, x):
#         x = self.affine1(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#         action_scores = self.affine2(x)
#         return F.softmax(action_scores, dim=1)



# policy = Policy(nb_states = env.codebook_size, window_len = args.window_length, num_measurements=args.num_measurements)
# optimizer = optim.Adam(policy.parameters(), lr=1e-3)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REINFORCE')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
    parser.add_argument('--render', default=False, help='render the environment')
    # parser.add_argument('--log-interval', type=int, default=50, metavar='N', help='interval between training status logs (default: 10)')
    parser.add_argument('--window_length', default=5, type=int, help='')
    parser.add_argument('--combine_state', default= False)
    parser.add_argument('--num_measurements', default=8,type=int)
    parser.add_argument('--rmsize', default=60000, type=int, help='memory size')
    parser.add_argument('--num_beams_per_UE',default=8,type=int)
    parser.add_argument('--enable_baseline',default=True)
    parser.add_argument('--enable_genie',default=True)
    parser.add_argument('--enable_exhaustive',default=True)
    parser.add_argument('--ue_speed',default=10)
    parser.add_argument('--full_observation', default=False)
    parser.add_argument('--oversampling_factor',type=int,default=1)
    parser.add_argument('--num_antennas',type=int,default=64)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--validate_episodes', default=100, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=1000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--train_iter', default=int(1e6+1), type=int, help='train iters each timestep')
    parser.add_argument('--debug', default = True, dest='debug')
        
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    env = BeamManagementEnv(num_antennas = args.num_antennas,
                            oversampling_factor = args.oversampling_factor, 
                            num_beams_per_UE = args.num_beams_per_UE,
                            ue_speed = args.ue_speed, 
                            enable_baseline=args.enable_baseline, 
                            enable_genie=args.enable_genie,
                            enable_exhaustive = args.enable_exhaustive,
                            combine_state=args.combine_state,
                            full_observation = args.full_observation,
                            num_measurements = args.num_measurements)
    eval_env = deepcopy(env)
        
    agent = REINFORCE_Agent(args)
    
    agent_rewards = []
    if env.enable_baseline:
        baseline_rewards = []
    if env.enable_genie:
        genie_rewards = []
    step = episode = episode_steps = 0
    episode_reward = 0.
    ob = None
    eval_rewards = []        
    ob = None
    
    # train_memory = BeamSpaceSequentialMemory(limit=args.rmsize, window_length=args.window_length, num_measurements=args.num_measurements)

    while step < args.train_iter:
        if ob is None:
            train_memory = BeamSpaceSequentialMemory(limit=args.rmsize, window_length=args.window_length, num_measurements=args.num_measurements)
            ob, info = deepcopy(env.reset())
        state = train_memory.get_recent_state(ob)
        action = agent.select_action(state)
        ob_next, reward, done, info = env.step(action)
        if args.max_episode_length and episode_steps >= args.max_episode_length -1:
            done = True
        train_memory.append(ob,action,reward,done)
        ob = deepcopy(ob_next)
        agent.policy.rewards.append(reward)
        episode_reward += reward
        
        #evaluation
        if args.validate_steps > 0 and step % args.validate_steps == 0:
            eval_agent_reward = []
            if eval_env.enable_baseline:
                eval_baseline_reward = []
            if eval_env.enable_genie:
                eval_genie_reward = []
            if eval_env.enable_exhaustive:
                eval_exhaustive_reward = []
                
            for val_epi in range(args.validate_episodes):
                eval_memory = BeamSpaceSequentialMemory(limit=args.rmsize, window_length=args.window_length, num_measurements=args.num_measurements)
                eval_ob, eval_info = deepcopy(eval_env.reset())
                eval_done = False
                while not eval_done:
                    eval_state = eval_memory.get_recent_state(eval_ob)
                    eval_action = agent.eval_select_action(eval_state)
                    eval_ob_next, eval_reward, eval_done, eval_info = eval_env.step(eval_action)
                    eval_memory.append(eval_ob,eval_action,eval_reward,eval_done)
                    eval_ob = eval_ob_next
                    
                eval_agent_reward.append(np.array(eval_env.reward_log['agent']).mean())
                if eval_env.enable_baseline:
                    eval_baseline_reward.append(np.array(eval_env.reward_log['baseline']).mean())
                if eval_env.enable_genie:
                    eval_genie_reward.append(np.array(eval_env.reward_log['genie']).mean())
                if eval_env.enable_exhaustive:
                    eval_exhaustive_reward.append(np.array(eval_env.reward_log['exhaustive']).mean())
            plt.figure()
            sns.kdeplot(eval_agent_reward,label='agent')
            if eval_env.enable_baseline:
                sns.kdeplot(eval_baseline_reward,label='baseline')
            if eval_env.enable_genie:
                sns.kdeplot(eval_genie_reward,label='upperbound')
            if eval_env.enable_exhaustive:
                sns.kdeplot(eval_exhaustive_reward,label='iterative selection with genie')
            plt.legend();
            # plt.figure()
            # plt.plot(rewards[0])
            # plt.xlabel('episodes')
            # plt.ylabel('avg episode reward')
            plt.title('Eval Results after #{} Training Steps'.format(step))
            plt.show()
            
        step += 1
        episode_steps += 1
        episode_reward += reward
        
        if done:       
            # if debug: prGreen('#{}: mean_episode_reward:{} episode_reward:{} steps:{}'.format(episode,episode_reward/episode_steps,episode_reward,step))
            episode_avg_agent_reward = sum(env.reward_log['agent'])/episode_steps
            agent_rewards.append(episode_avg_agent_reward)
            if env.enable_baseline:
                episode_avg_baseline_reward = sum(env.reward_log['baseline'])/episode_steps
                baseline_rewards.append(episode_avg_baseline_reward)
            if env.enable_genie:
                episode_avg_genie_reward = sum(env.reward_log['genie'])/episode_steps
                genie_rewards.append(episode_avg_genie_reward)            
        
            if args.debug:                
                if env.enable_baseline and env.enable_genie:
                    print('Episode #{:5d} {:5d} steps: agent:{:07.4f} baseline:{:07.4f} genie:{:07.4f}'.format(episode,episode_steps,episode_avg_agent_reward,episode_avg_baseline_reward,episode_avg_genie_reward))
                elif env.enable_baseline:
                    print('Episode #{:5d}: agent:{:07.4f} baseline:{:07.4f}'.format(episode,episode_avg_agent_reward,episode_avg_baseline_reward))
                elif env.enable_genie:
                    print('Episode #{:5d}: agent:{:07.4f} genie:{:07.4f}'.format(episode,episode_avg_agent_reward,episode_avg_genie_reward))
                else:
                    print('Episode #{:5d}: agent:{:07.4f}'.format(episode,episode_avg_agent_reward))
                    
            # reset
            agent.finish_episode()
            ob = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            
        # ep_reward = 0
        # ob, info = env.reset()
        # for t in range(1, 10000):  # Don't infinite loop while learning
        #     state = memory.get_recent_state(ob)
        #     action = select_action(state)
        #     ob_next, reward, done, info = env.step(action)
        #     memory.append(ob, action, reward,done)
        #     ob = ob_next
        #     # if args.render:
        #     #     env.render()
        #     policy.rewards.append(reward)
        #     ep_reward += reward
        #     if done:
        #         break
        # # running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        # finish_episode()
        # train_rewards.append(ep_reward/t)
        # if i_episode % args.log_interval == 0:
        #     print('Episode {}\tLast reward: {:.2f}'.format(i_episode, ep_reward/t))
            