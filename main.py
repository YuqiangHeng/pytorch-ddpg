#!/usr/bin/env python3 

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym

from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG
from multiwindow_DDPG import multiwindow_DDPG
from util import *
from BeamManagementEnv import BeamManagementEnv
import matplotlib.pyplot as plt

# gym.undo_logger_setup()
    
def train(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False):
    
    agent_rewards = []
    if env.enable_baseline:
        baseline_rewards = []
    if env.enable_genie:
        genie_rewards = []
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None

    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)
        
        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup :
            agent.update_policy()
        
        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            val_env = deepcopy(env)
            validate_reward = evaluate(val_env, policy, debug=False, visualize=False)
            # if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
            print('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

        # # [optional] save intermideate model
        # if step % int(num_iterations/100) == 0:
        #     agent.save_model(output)

        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done: # end of episode
            # if debug: prGreen('#{}: mean_episode_reward:{} episode_reward:{} steps:{}'.format(episode,episode_reward/episode_steps,episode_reward,step))
            episode_avg_agent_reward = sum(env.reward_log['agent'])/episode_steps
            agent_rewards.append(episode_avg_agent_reward)
            if env.enable_baseline:
                episode_avg_baseline_reward = sum(env.reward_log['baseline'])/episode_steps
                baseline_rewards.append(episode_avg_baseline_reward)
            if env.enable_genie:
                episode_avg_genie_reward = sum(env.reward_log['genie'])/episode_steps
                genie_rewards.append(episode_avg_genie_reward)            
        
            if debug:
                
                if env.enable_baseline and env.enable_genie:
                    print('#{:5d}: agent:{:07.4f} baseline:{:07.4f} genie:{:07.4f}'.format(episode,episode_avg_agent_reward,episode_avg_baseline_reward,episode_avg_genie_reward))
                elif env.enable_baseline:
                    print('#{:5d}: agent:{:07.4f} baseline:{:07.4f}'.format(episode,episode_avg_agent_reward,episode_avg_baseline_reward))
                elif env.enable_genie:
                    print('#{:5d}: agent:{:07.4f} genie:{:07.4f}'.format(episode,episode_avg_agent_reward,episode_avg_genie_reward))
                else:
                    print('#{:5d}: agent:{:07.4f}'.format(episode,episode_avg_agent_reward))


            # temp_action = agent.select_action(observation)
            # agent.memory.append(
            #     np.concatenate((observation, temp_action),axis=0),
            #     temp_action,
            #     0., False
            # )
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
    if env.enable_baseline and env.enable_genie:
        return [agent_rewards, baseline_rewards, genie_rewards]
    elif env.enable_baseline:
        return [agent_rewards, baseline_rewards]
    elif env.enable_genie:
        return [agent_rewards, genie_rewards]
    else:
        return agent_rewards

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))

import time
import seaborn as sns

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='beam', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=0, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    # parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=10000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    
    parser.add_argument('--window_length', default=5, type=int, help='')
    parser.add_argument('--combine_state', default= False)
    parser.add_argument('--num_measurements', default=5,type=int)
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO
    parser.add_argument('--num_beams_per_UE',default=8,type=int)
    parser.add_argument('--enable_baseline',default=True)
    parser.add_argument('--enable_genie',default=True)
    parser.add_argument('--ue_speed',default=10)
    parser.add_argument('--full_observation', default=False)
    parser.add_argument('--conv2d_1_kernel_size',type=int,default=5)
    parser.add_argument('--conv2d_2_kernel_size',type=int,default=3)
    parser.add_argument('--oversampling_factor',type=int,default=1)
    parser.add_argument('--num_antennas',type=int,default=8)
    parser.add_argument('--use_saved_traj_in_validation',default=True)
    
    parser.add_argument('--debug', default = False, dest='debug')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    # env = NormalizedEnv(gym.make(args.env))
    # env = BeamManagementEnv(enable_baseline = True, enable_genie = True)
    env = BeamManagementEnv(num_antennas = args.num_antennas,
                            oversampling_factor = args.oversampling_factor, 
                            num_beams_per_UE = args.num_beams_per_UE,
                            ue_speed = args.ue_speed, 
                            enable_baseline=args.enable_baseline, 
                            enable_genie=args.enable_genie,
                            combine_state=args.combine_state,
                            full_observation = args.full_observation,
                            num_measurements = args.num_measurements)
    # env = BeamManagementEnvMultiFrame(window_length = window_len, enable_baseline=True,enable_genie=True)
    
    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]


    # agent = DDPG(nb_states, nb_actions, window_len, args)
    agent = multiwindow_DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(num_episodes = args.validate_episodes, 
                         interval = args.validate_steps, 
                         save_path = args.output, 
                         max_episode_length=args.max_episode_length,
                         use_saved_traj= args.use_saved_traj_in_validation)
    
    
    if args.mode == 'train':
        tic = time.time()
        rewards = train(args.train_iter, agent, env, evaluate, 
            args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)
        toc = time.time()
        print('Training time for {} steps = {} seconds'.format(args.train_iter, toc-tic))
        if args.enable_baseline and args.enable_genie:
            plt.figure()
            sns.kdeplot(rewards[0],label='agent')
            sns.kdeplot(rewards[1],label='baseline')
            sns.kdeplot(rewards[2],label='genie')
            plt.legend();
            plt.figure()
            plt.plot(rewards[0])
            plt.xlabel('episodes')
            plt.ylabel('avg episode reward')
        elif args.enable_baseline:
            plt.figure()
            sns.kdeplot(rewards[0],label='agent')
            sns.kdeplot(rewards[1],label='baseline')
            plt.legend();
            plt.figure()
            plt.plot(rewards[0])
            plt.xlabel('episodes')
            plt.ylabel('avg episode reward')
        elif args.enable_genie:
            plt.figure()
            sns.kdeplot(rewards[0],label='agent')
            sns.kdeplot(rewards[2],label='genie')
            plt.legend();
            plt.figure()
            plt.plot(rewards[0])
            plt.xlabel('episodes')
            plt.ylabel('avg episode reward')
        else:
            plt.figure()
            sns.kdeplot(rewards,label='agent')
            plt.legend();
            plt.figure()
            plt.plot(rewards)
            plt.xlabel('episodes')
            plt.ylabel('avg episode reward')

        
            
                    

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
