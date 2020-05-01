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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Multinomial
from BeamManagementEnv import BeamManagementEnv
import time

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', default=False,
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')

parser.add_argument('--window_length', default=5, type=int, help='')
parser.add_argument('--combine_state', default= False)
parser.add_argument('--num_measurements', default=8,type=int)
parser.add_argument('--rmsize', default=60000, type=int, help='memory size')
parser.add_argument('--num_beams_per_UE',default=8,type=int)
parser.add_argument('--enable_baseline',default=False)
parser.add_argument('--enable_genie',default=False)
parser.add_argument('--ue_speed',default=10)
parser.add_argument('--full_observation', default=False)
parser.add_argument('--oversampling_factor',type=int,default=1)
parser.add_argument('--num_antennas',type=int,default=64)
parser.add_argument('--use_saved_traj_in_validation',default=False)
parser.add_argument('--actor_lambda',type=float,default=0.5)

parser.add_argument('--debug', default = False, dest='debug')
    
args = parser.parse_args()


env = BeamManagementEnv(num_antennas = args.num_antennas,
                        oversampling_factor = args.oversampling_factor, 
                        num_beams_per_UE = args.num_beams_per_UE,
                        ue_speed = args.ue_speed, 
                        enable_baseline=args.enable_baseline, 
                        enable_genie=args.enable_genie,
                        combine_state=args.combine_state,
                        full_observation = args.full_observation,
                        num_measurements = args.num_measurements)
    
torch.manual_seed(args.seed)


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

class Policy(nn.Module):
    def __init__(self, nb_states, window_len, num_measurements):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_shape = (window_len*num_measurements, nb_states)
        nsize = self._serial_input_size(self.input_shape)
        hidden1 = int(nsize/2)
        hidden2 = int(nsize/4)
        hidden3 = int(nsize/8)
        self.outdim = nb_states
        self.outshape = nb_states
        self.saved_log_probs = []
        self.rewards = []
        self.fc1 = nn.Sequential(nn.Linear(nsize,hidden1),
                                    nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden1,hidden2),
                                    nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(hidden2,hidden2),
                                    nn.ReLU())
        # self.fc4 = nn.Sequential(nn.Linear(hidden2,hidden2),
        #                             nn.ReLU())
        self.fc5 = nn.Sequential(nn.Linear(hidden2,self.outdim),
                                    nn.ReLU())
        self.fc6 = nn.Linear(self.outdim,self.outdim)

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

def sample_topk(probs,k):
    m = Categorical(probs)
    sel = []
    K = 0
    while len(np.unique(sel)) < k:
        b = m.sample()
        sel.append(b.item())
        K += 1
    action = np.unique(sel)
    unselected = [i for i in range(len(probs)) if not i in action]
    # p_action = 0
    # for i in unselected:
    #     p_action += (1-probs[i].item())**K
    p_action = (1-probs[unselected].sum())**K
    log_p_action = torch.log(p_action)
    binary_action = np.zeros(len(probs))
    binary_action[action] = 1
    return binary_action, log_p_action
    

policy = Policy(nb_states = env.codebook_size, window_len = args.window_length, num_measurements=args.num_measurements)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state).squeeze()
    # tic = time.time()
    action, log_p_action = sample_topk(probs,args.num_beams_per_UE)
    # toc = time.time()
    # print('Time taken to sample top-K = {} seconds'.format(toc-tic))
    # m = Categorical(probs)
    # action = m.sample()
    # policy.saved_log_probs.append(m.log_prob(action))
    policy.saved_log_probs.append(log_p_action)
    return action


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]



def main():
    running_reward = 10
    for i_episode in range(1000):
        memory = BeamSpaceSequentialMemory(limit=args.rmsize, window_length=args.window_length, num_measurements=args.num_measurements)
        ep_reward = 0
        ob, info = env.reset()
        for t in range(1, 10000):  # Don't infinite loop while learning
            state = memory.get_recent_state(ob)
            action = select_action(state)
            ob_next, reward, done, info = env.step(action)
            memory.append(ob, action, reward,done)
            ob = ob_next
            # if args.render:
            #     env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break


if __name__ == '__main__':
    main()