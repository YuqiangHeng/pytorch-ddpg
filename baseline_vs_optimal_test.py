# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 19:24:19 2020

@author: ethan
"""

import numpy as np
from BeamManagementEnv import BeamManagementEnv
import matplotlib.pyplot as plt

#plot sample trajactory
env = BeamManagementEnv(ue_speed = 5, enable_baseline =False, enable_genie=False)
env.reset()
plt.figure()
plt.plot(env.ue_loc[env.traj,0],env.ue_loc[env.traj,1])
plt.xlabel('meters')
plt.ylabel('meters')
plt.title('sample trajectory')
plt.savefig('sample_traj.png')

#plot genie vs baseline vs ue speed
speeds = np.arange(1,30,5)
niter = int(1e3)
genie_rewards = np.zeros((len(speeds),niter))
baseline_rewards = np.zeros((len(speeds),niter))
for v_i,v in enumerate(speeds):
    env = BeamManagementEnv(ue_speed = v, enable_baseline =True, enable_genie=True)
    env.reset()
    for step in range(niter):
        random_action = np.random.rand(64)
        s,r,done,info = env.step(random_action)
        genie_rewards[v_i,step] = info['genie_reward']
        baseline_rewards[v_i,step] = info['baseline_reward']
        if done:
            env.reset()

plt.figure()
plt.plot(speeds,[baseline_rewards[i,:].mean() for i in range(len(speeds))],label='baseline')
plt.plot(speeds,[genie_rewards[i,:].mean() for i in range(len(speeds))],label='genie')
plt.legend()
plt.xlabel('distance travelled between beam set configuration')
plt.ylabel('mean spectral efficiency')

speeds = np.arange(1,30,5)
niter = int(1e3)
genie_beamset_size = np.zeros((len(speeds),niter))
for v_i,v in enumerate(speeds):
    env = BeamManagementEnv(ue_speed = v, enable_baseline =False, enable_genie=True)
    env.reset()
    for step in range(niter):
        random_action = np.random.rand(64)
        s,r,done,info = env.step(random_action)
        genie_beamset_size[v_i,step] = len(info['genie_beams'])
        if done:
            env.reset()

plt.figure()
plt.plot(speeds,[baseline_rewards[i,:].mean() for i in range(len(speeds))],label='baseline')
plt.legend()
plt.xlabel('distance travelled between beam set configuration')
plt.ylabel('mean spectral efficiency')

for i,v in enumerate(speeds):
    plt.figure()
    plt.hist(genie_beamset_size[i,:])
    plt.xlabel('number of unique beams configured')
    title = 'beam configuration interval: {} m'.format(v)
    savefname = 'beamset_size_hist_beam_configuration_interval_{}_m.png'.format(v)
    plt.title(title)
    plt.savefig(savefname)
# import time
# niter = int(1e3)
# v = 50
# env = BeamManagementEnv(ue_speed = v, enable_baseline =True, enable_genie=True)
# env.reset()
# tic = time.perf_counter()
# for step in range(niter):
#     random_action = np.random.rand(64)
#     s,r,done,info = env.step(random_action)
#     if done:
#         env.reset()
# toc = time.perf_counter()
# # prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
# print(f"both baselines in {toc - tic:0.4f} seconds")

# env = BeamManagementEnv(ue_speed = v, enable_baseline =False, enable_genie=False)
# env.reset()
# tic = time.perf_counter()
# for step in range(niter):
#     random_action = np.random.rand(64)
#     s,r,done,info = env.step(random_action)
#     if done:
#         env.reset()
# toc = time.perf_counter()
# print(f"neither baselines in {toc - tic:0.4f} seconds")