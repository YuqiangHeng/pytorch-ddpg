
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from memory import BeamSpaceSequentialMemory
from util import *
from Autoencoder_DDPG import Autoencoder_DDPG
from SubActionDDPG import SubAction_DDPG
from copy import deepcopy
import torch

class Evaluator(object):

    def __init__(self, num_episodes, interval, save_path='', max_episode_length=None, use_saved_traj=False):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes,0)
        self.use_saved_traj = use_saved_traj

    def __call__(self, env, policy, debug=False, visualize=False, save=False):
            
        if self.use_saved_traj:
            env.set_data_mode(use_saved_trajectory = self.use_saved_traj, num_saved_traj = self.num_episodes)
        else:
            env.set_data_mode(use_saved_trajectory = self.use_saved_traj)
        self.is_training = False
        observation = None
        result = []

        for episode in range(self.num_episodes):

            # reset at the start of episode
            observation = env.reset()
            episode_steps = 0
            episode_reward = 0.
                
            assert observation is not None

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(observation)

                observation, reward, done, info = env.step(action)
                if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                    done = True
                
                if visualize:
                    # env.render(mode='human')
                    env.render()

                # update
                episode_reward += reward
                episode_steps += 1

            if debug: print('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            result.append(episode_reward)

        result = np.array(result).reshape(-1,1)
        self.results = np.hstack([self.results, result])

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        
        # allow env to generate traj
        env.set_data_mode(use_saved_trajectory = False, num_saved_traj = None)
        
        return np.mean(result)

    def save_results(self, fn):

        y = np.mean(self.results, axis=0)
        error=np.std(self.results, axis=0)
                    
        x = range(0,self.results.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'.png')
        savemat(fn+'.mat', {'reward':self.results})
                
class DDPGAgentEval(object):
    def __init__(self, agent:Autoencoder_DDPG):
        self.seed(agent.seed)
        self.nb_states = agent.nb_states
        self.nb_actions= agent.nb_actions
        self.num_beams_per_UE = agent.num_beams_per_UE
        self.actor = deepcopy(agent.actor)
        self.actor.eval()
        self.memory = BeamSpaceSequentialMemory(limit = int(1e6), window_length = agent.window_length, num_measurements = agent.num_measurements)
        self.ob_t = None
        self.a_t = None
        if USE_CUDA: self.actor.cuda()
    
    def observe(self, r_t, ob_t1, done):
        self.memory.append(self.ob_t, self.a_t, r_t, done)
        self.ob_t = ob_t1

    def select_action(self, observation):
        s_t = self.memory.get_recent_state(observation)
        #remove existing empty dimension and add batch dimension 
        s_t_array = np.array([np.squeeze(np.array(s_t))])
        # actor_output = to_numpy(self.actor(torch.from_numpy(s_t_array))).squeeze(0)
        actor_output = to_numpy(self.actor(to_tensor(s_t_array)))
        action = self.actor.select_beams(actor_output, self.nb_actions, self.num_beams_per_UE) 
        action = action.squeeze(0)
        self.a_t = action
        return action
    
    def pick_beams(self, observation:np.ndarray):
        #observation is num_measurements x num_beams matrix, iteratively pick best beam
        selected_beams = np.argsort(np.sum(observation,axis=0))[-self.num_beams_per_UE:]
        binary_beams = np.zeros(self.nb_actions)
        binary_beams[selected_beams] = 1
        return binary_beams
    
    def reset(self, obs):
        self.ob_t = obs
 
    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
            
class BeamSelectionEvaluator(object):

    def __init__(self, num_episodes, interval, save_path='', max_episode_length=None, use_saved_traj=False):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes,0)
        self.use_saved_traj = use_saved_traj

    def __call__(self, env, agent: DDPGAgentEval, debug=False, visualize=False, save=False):
        agent_rewards = []
        if env.enable_baseline:
            baseline_rewards = []
        if env.enable_genie:
            genie_rewards = []
        if env.enable_exhaustive:
            exhaustive_rewards = []
        step = episode = episode_steps = 0
        episode_reward = 0.
        observation = None          
        
        if self.use_saved_traj:
            env.set_data_mode(use_saved_trajectory = self.use_saved_traj, num_saved_traj = self.num_episodes)
        else:
            env.set_data_mode(use_saved_trajectory = self.use_saved_traj)
        self.is_training = False
                        
        while episode < self.num_episodes:
            if observation is None:
                # reset at the start of episode
                observation,info = deepcopy(env.reset())
                agent.reset(observation)
                episode_steps = 0
                episode_reward = 0.
            # start episode
            action = agent.select_action(observation)
            observation2, reward, done, info = env.step(action)
            observation2 = deepcopy(observation2)
            if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                done = True
            
            if isinstance(agent, SubAction_DDPG):
                agent.observe(info['beams_spe'], reward, observation2, done)
            else:
                agent.observe(reward, observation2, done)
        
            #update
            step += 1
            episode_steps += 1
            episode_reward += reward
            observation = deepcopy(observation2)

            if done: # end of episode
                episode_avg_agent_reward = sum(env.reward_log['agent'])/episode_steps
                agent_rewards.append(episode_avg_agent_reward)
                if env.enable_baseline:
                    episode_avg_baseline_reward = sum(env.reward_log['baseline'])/episode_steps
                    baseline_rewards.append(episode_avg_baseline_reward)
                if env.enable_genie:
                    episode_avg_genie_reward = sum(env.reward_log['genie'])/episode_steps
                    genie_rewards.append(episode_avg_genie_reward)    
                if env.enable_exhaustive:
                    episode_avg_exhaustive_reward = sum(env.reward_log['exhaustive'])/episode_steps
                    exhaustive_rewards.append(episode_avg_exhaustive_reward)
            
                if debug:
                    if env.enable_baseline and env.enable_genie:
                        print('Eval #{:5d}: agent:{:07.4f} baseline:{:07.4f} genie:{:07.4f}'.format(episode,episode_avg_agent_reward,episode_avg_baseline_reward,episode_avg_genie_reward))
                    elif env.enable_baseline:
                        print('Eval #{:5d}: agent:{:07.4f} baseline:{:07.4f}'.format(episode,episode_avg_agent_reward,episode_avg_baseline_reward))
                    elif env.enable_genie:
                        print('Eval #{:5d}: agent:{:07.4f} genie:{:07.4f}'.format(episode,episode_avg_agent_reward,episode_avg_genie_reward))
                    else:
                        print('Eval #{:5d}: agent:{:07.4f}'.format(episode,episode_avg_agent_reward))
    
                agent.memory.append(
                    observation,
                    agent.select_action(observation),
                    0., False
                )
                # reset
                observation = None
                episode_steps = 0
                episode_reward = 0.
                episode += 1

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        
        # allow env to generate traj
        env.set_data_mode(use_saved_trajectory = False, num_saved_traj = None)
                
        return_results = {'agent_rewards':agent_rewards}
        if env.enable_baseline:
            return_results['baseline_rewards'] = baseline_rewards
        if env.enable_genie:
            return_results['genie_rewards'] = genie_rewards
        if env.enable_exhaustive:
            return_results['exhaustive_rewards'] = exhaustive_rewards
            
            
        # if env.enable_baseline and env.enable_genie:
        #     return {'agent_rewards':agent_rewards, 'baseline_rewards':baseline_rewards, 'genie_rewards':genie_rewards}
        # elif env.enable_baseline:
        #     return {'agent_rewards':agent_rewards, 'baseline_rewards':baseline_rewards}
        # elif env.enable_genie:
        #     return {'agent_rewards':agent_rewards, 'genie_rewards':genie_rewards}
        # else:
        #     return {'agent_rewards':agent_rewards}
        
        return return_results

    def save_results(self, fn):

        y = np.mean(self.results, axis=0)
        error=np.std(self.results, axis=0)
                    
        x = range(0,self.results.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'.png')
        savemat(fn+'.mat', {'reward':self.results})