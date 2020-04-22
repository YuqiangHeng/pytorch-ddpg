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

def pick_beams(observation:np.ndarray, num_beams_per_UE, nb_actions):
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
    
    
if __name__ == "__main__":

    env = BeamManagementEnv(num_antennas = 64,
                            oversampling_factor = 1, 
                            num_beams_per_UE = 8,
                            ue_speed = 15, 
                            enable_baseline = True, 
                            enable_genie = True,
                            enable_exhaustive = True,
                            combine_state = False,
                            full_observation = True,
                            num_measurements = 8)
    
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    
    num_epoch = int(1e3)
    baseline = []
    genie = []
    exhaustive = []
    test = []
    
    for epi in tqdm(range(num_epoch)):        
        done = False
        baseline_epi = genie_epi = exhaustive_epi = test_epi = epi_step = 0
        ob = env.reset()
        prev_ob = ob
        while not done:    
            action = pick_beams(ob, 8, nb_actions)
            # action = np.zeros(nb_actions)
            # action[np.argsort(ob)[-8:]] = 1
            ob, r, done, info = env.step(action)
            baseline_epi += info['baseline_reward']
            genie_epi += info['genie_reward']
            exhaustive_epi += info['incremental_reward']
            test_epi += r
            epi_step += 1
        baseline.append(baseline_epi/epi_step)
        genie.append(genie_epi/epi_step)
        exhaustive.append(exhaustive_epi/epi_step)
        test.append(test_epi/epi_step)
    
    plt.figure()
    sns.kdeplot(test,label='iterative selection with perfect predictor')
    sns.kdeplot(baseline,label='baseline')
    sns.kdeplot(genie,label='upperbound')
    sns.kdeplot(exhaustive,label='iterative selection with genie')
    plt.legend();    
    