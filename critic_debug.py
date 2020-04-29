# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:36:15 2020

@author: ethan
"""


import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic,MLP,SerializedCritic)
from memory import BeamSpaceSequentialMemory
from util import *
import pickle
import argparse
from copy import deepcopy

from evaluator import Evaluator, BeamSelectionEvaluator, DDPGAgentEval
from BeamManagementEnv import BeamManagementEnv
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import seaborn as sns

def top_k_selection(observation:np.ndarray, num_beams_per_UE, nb_actions):
    #observation is num_measurements x num_beams matrix, iteratively pick best beam
    # selected_beams = np.argsort(np.sum(observation,axis=0))[-num_beams_per_UE:]
    
    selected_beams = np.argsort(np.sum(observation,axis=0))[-num_beams_per_UE:]
    binary_beams = np.zeros(nb_actions)
    binary_beams[np.array(selected_beams)] = 1        
    return binary_beams

def worst_k_selection(observation:np.ndarray, num_beams_per_UE, nb_actions):
    #observation is num_measurements x num_beams matrix, iteratively pick best beam
    # selected_beams = np.argsort(np.sum(observation,axis=0))[-num_beams_per_UE:]
    
    selected_beams = np.argsort(np.sum(observation,axis=0))[0:num_beams_per_UE]
    binary_beams = np.zeros(nb_actions)
    binary_beams[np.array(selected_beams)] = 1        
    return binary_beams

if __name__ == "__main__":
    agent = pickle.load(open('debug_saved_agent.pkl','rb'))
    env = pickle.load(open('debug_saved_env.pkl','rb'))
    env.full_observation = True
    q_best_dist = []
    q_worst_dist = []
    # for epi in range(100):
    #     ob,info = env.reset()
    #     agent.reset(ob)
    #     done = False
    #     while not done:
    #         best_action = np.array([top_k_selection(info['next_observation'], env.num_beams_per_UE, env.codebook_size)])
    #         worst_action = np.array([worst_k_selection(info['next_observation'], env.num_beams_per_UE, env.codebook_size)])
    #         s_t = np.array([agent.memory.get_recent_state(ob)])
    #         q_best = agent.critic([to_tensor(s_t),to_tensor(best_action)])
    #         q_worst= agent.critic([to_tensor(s_t),to_tensor(worst_action)])
    #         q_best_dist.append(q_best.item())
    #         q_worst_dist.append(q_worst.item())
    #         ob, r, done, info = env.step(best_action)
    #         agent.observe(r,ob,done)
            
    state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = agent.memory.sample_and_split(1000)

    # Prepare for the target q batch
    with torch.no_grad():
        next_states = to_tensor(next_state_batch)
        target_actor_output = self.actor_target(next_states) 
        # next_actions = torch.from_numpy(self.pick_beams_batch(to_numpy(predicted_beam_qual_target)))
        next_actions = to_tensor(self.actor_target.select_beams(to_numpy(target_actor_output), self.nb_actions, self.num_beams_per_UE))            
        next_q_values = self.critic_target([next_states,next_actions])
    # next_q_values.volatile=False
    next_q_values.requires_grad = True
    target_q_batch = to_tensor(reward_batch) + \
        self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

    # Critic update
    self.critic.zero_grad()
    q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
    value_loss = criterion(q_batch, target_q_batch)
    # print(value_loss.item())
    value_loss.backward()
    self.training_log['critic_mse'].append(value_loss.item())
    self.critic_optim.step()

    # Actor update
    self.actor.zero_grad()
    
    # Beam qual prediction loss, only if using MSE of actor output
    states = to_tensor(state_batch)
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
    actions = to_tensor(self.actor.select_beams(to_numpy(actor_output),self.nb_actions,self.num_beams_per_UE))        
    policy_loss = -self.critic([states,actions])
    policy_loss = policy_loss.mean()
    self.training_log['actor_value'].append(policy_loss.item())

    plt.figure()
    sns.kdeplot(q_best_dist,label='q best')
    sns.kdeplot(q_worst_dist,label='q worst ')
    plt.legend();  
    
            