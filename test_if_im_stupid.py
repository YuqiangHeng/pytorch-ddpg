# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:30:14 2020

@author: ethan
"""

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym

from normalized_env import NormalizedEnv
from evaluator import Evaluator, BeamSelectionEvaluator, DDPGAgentEval
from ddpg import DDPG
from multiwindow_DDPG import multiwindow_DDPG
from Autoencoder_DDPG import Autoencoder_DDPG
from util import *
from BeamManagementEnv import BeamManagementEnv
import matplotlib.pyplot as plt
import time
import seaborn as sns
from tqdm import tqdm

def iterative_selection(observation:np.ndarray, num_beams_per_UE, nb_actions):
    #observation is num_measurements x num_beams matrix, iteratively pick best beam
    # selected_beams = np.argsort(np.sum(observation,axis=0))[-num_beams_per_UE:]
    
    selected_beams = []
    pool = list(np.arange(nb_actions))
    while len(selected_beams) < num_beams_per_UE:
        temp_r = [observation[:,np.array(selected_beams+[i])].max(axis=1).sum() for i in pool]
        temp_b = pool[np.argmax(temp_r)]
        selected_beams.append(temp_b)
        pool.remove(temp_b)
    
    binary_beams = np.zeros(nb_actions)
    binary_beams[np.array(selected_beams)] = 1        
    return binary_beams

def top_k_selection(observation:np.ndarray, num_beams_per_UE, nb_actions):
    #observation is num_measurements x num_beams matrix, iteratively pick best beam
    # selected_beams = np.argsort(np.sum(observation,axis=0))[-num_beams_per_UE:]
    
    selected_beams = np.argsort(np.sum(observation,axis=0))[-num_beams_per_UE:]
    binary_beams = np.zeros(nb_actions)
    binary_beams[np.array(selected_beams)] = 1        
    return binary_beams
    
    
if __name__ == "__main__":

    env = BeamManagementEnv(num_antennas = 64,
                            oversampling_factor = 1, 
                            num_beams_per_UE = 8,
                            ue_speed = 10, 
                            enable_baseline = True, 
                            enable_genie = True,
                            enable_exhaustive = True,
                            combine_state = False,
                            full_observation = True,
                            num_measurements = 8)
    
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    
    num_epoch = int(5e2)
    baseline = []
    genie = []
    iterative = []
    random = []
    test = []
    top_k = []
    top_k_genie = []
    top_k_gap = []
    
    for epi in tqdm(range(num_epoch)):        
        done = False
        baseline_epi = genie_epi = iterative_epi = test_epi = random_epi = top_k_epi = top_k_genie_epi = epi_step = 0
        # ob = env.reset()
        ob, info = env.reset()
        # prev_ob = ob
        while not done:    
            print(env.ue_traveled_dist_next)
            action = iterative_selection(info['next_observation'], env.num_beams_per_UE, nb_actions)
            random_beams = np.random.choice(np.arange(env.codebook_size),env.num_beams_per_UE,replace=False)
            random_epi += env._calc_reward(random_beams)[0]
            top_k_genie_r = env._calc_reward(np.nonzero(top_k_selection(info['next_observation'],env.num_beams_per_UE,nb_actions))[0])[0]
            top_k_genie_epi += top_k_genie_r
            top_k_epi += env._calc_reward(np.nonzero(top_k_selection(ob,env.num_beams_per_UE,nb_actions))[0])[0]
            # action = np.zeros(nb_actions)
            # action[np.argsort(ob)[-8:]] = 1
            ob, r, done, info = env.step(action)
            baseline_epi += info['baseline_reward']
            genie_epi += info['genie_reward']
            iterative_epi += info['incremental_reward']
            top_k_gap.append(info['genie_reward']-top_k_genie_r)
            if info['genie_reward']-top_k_genie_r < 0:
                print('ahh')
            test_epi += r
            epi_step += 1
        baseline.append(baseline_epi/epi_step)
        genie.append(genie_epi/epi_step)
        iterative.append(iterative_epi/epi_step)
        random.append(random_epi/epi_step)
        test.append(test_epi/epi_step)
        top_k_genie.append(top_k_genie_epi/epi_step)
        top_k.append(top_k_epi/epi_step)
    
    plt.figure()
    sns.kdeplot(test,label='iterative selection with perfect predictor')
    sns.kdeplot(baseline,label='baseline')
    sns.kdeplot(genie,label='upperbound')
    sns.kdeplot(iterative,label='iterative selection with genie')
    sns.kdeplot(random,label='random selection')
    sns.kdeplot(top_k,label='top k selection')
    sns.kdeplot(top_k_genie,label='top k selection with genie')
    plt.legend();  
    
    plt.figure()
    sns.kdeplot(top_k_gap,label='top_k_gap')

    