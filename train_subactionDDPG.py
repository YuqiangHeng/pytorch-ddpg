#!/usr/bin/env python3 

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym

from normalized_env import NormalizedEnv
from evaluator import Evaluator, BeamSelectionEvaluator, DDPGAgentEval
from ddpg import DDPG
from multiwindow_DDPG import multiwindow_DDPG
from SubActionDDPG import SubAction_DDPG
from util import *
from BeamManagementEnv import BeamManagementEnv
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

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
    if evaluate is not None:
        eval_rewards = []
        
    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation, info = deepcopy(env.reset())
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
        agent.observe(info['beams_spe'], reward, observation2, done)
        if step > args.warmup :
            agent.update_policy()
        
        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            # policy = lambda x: agent.select_action(x, decay_epsilon=False)
            tic = time.time()
            evalagent = DDPGAgentEval(agent)
            val_env = deepcopy(env)
            val_env.enable_baseline = True
            val_env.enable_genie = True
            val_env.enable_exhaustive = False
            if step == num_iterations-1:
                val_env.enable_exhaustive = True
            val_env.reset()
            validate_rewards = evaluate(val_env, evalagent, debug=False, visualize=False, save = False)
            toc = time.time()
            # if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
            print('[Evaluate] Step_{:07d}: time:{} seconds, mean_reward:{}'.format(step, toc-tic, np.mean(validate_rewards['agent_rewards'])))
            eval_rewards.append(np.mean(validate_rewards['agent_rewards']))
            plt.figure()
            sns.kdeplot(validate_rewards['agent_rewards'],label='agent')
            if val_env.enable_baseline:
                sns.kdeplot(validate_rewards['baseline_rewards'],label='baseline')
            if val_env.enable_genie:
                sns.kdeplot(validate_rewards['genie_rewards'],label='upperbound')
            if val_env.enable_exhaustive:
                sns.kdeplot(validate_rewards['exhaustive_rewards'],label='iterative selection with genie')
            plt.legend();
            # plt.figure()
            # plt.plot(rewards[0])
            # plt.xlabel('episodes')
            # plt.ylabel('avg episode reward')
            plt.title('Eval Results after #{} Training Steps'.format(step))
            plt.show()

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
                    print('Episode #{:5d} {:5d} steps: agent:{:07.4f} baseline:{:07.4f} genie:{:07.4f}'.format(episode,episode_steps,episode_avg_agent_reward,episode_avg_baseline_reward,episode_avg_genie_reward))
                elif env.enable_baseline:
                    print('Episode #{:5d} {:5d} steps: agent:{:07.4f} baseline:{:07.4f}'.format(episode,episode_steps,episode_avg_agent_reward,episode_avg_baseline_reward))
                elif env.enable_genie:
                    print('Episode #{:5d} {:5d} steps: agent:{:07.4f} genie:{:07.4f}'.format(episode,episode_steps,episode_avg_agent_reward,episode_avg_genie_reward))
                else:
                    print('Episode #{:5d} {:5d} steps: agent:{:07.4f}'.format(episode,episode_steps,episode_avg_agent_reward))
                
                if step - 1 > args.warmup and episode % 25 == 0:
                    plt.figure()
                    plt.plot(np.array(agent.training_log['critic_mse'])[:,0], label='total value mse')
                    plt.plot(np.array(agent.training_log['critic_mse'])[:,1], label='subaction value mse')
                    plt.plot(agent.training_log['actor_value'], label='actor value')
                    plt.ylabel('loss')
                    plt.xlabel('number of training iterations')
                    plt.title('actor/critic loss after {} steps'.format(step))
                    plt.legend()
                    plt.show()


            # temp_action = agent.select_action(observation)
            # agent.memory.append(
            #     np.concatenate((observation, temp_action),axis=0),
            #     temp_action,
            #     0., False
            # )
            agent.memory.append(
                observation,
                agent.select_action(observation),
                info['beams_spe'],
                0., False
            )
            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
    all_results = {'agent_rewards':agent_rewards}
    if evaluate is not None:
        all_results['evalulation_rewards'] = eval_rewards
    if env.enable_baseline:
        all_results['baseline_rewards'] = baseline_rewards
    if env.enable_genie:
        all_results['genie_rewards'] = genie_rewards
    # if env.enable_baseline and env.enable_genie:
    #     return {'agent_rewards':agent_rewards, 'baseline_rewards':baseline_rewards, 'genie_rewards':genie_rewards}
    # elif env.enable_baseline:
    #     return {'agent_rewards':agent_rewards, 'baseline_rewards':baseline_rewards}
    # elif env.enable_genie:
    #     return {'agent_rewards':agent_rewards, 'genie_rewards':genie_rewards}
    # else:
    #     return {'agent_rewards':agent_rewards}
    return all_results

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
    parser.add_argument('--warmup', default=128, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.9, type=float, help='')
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    # parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--rmsize', default=60000, type=int, help='memory size')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=100, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=100, type=int, help='')
    parser.add_argument('--validate_steps', default=5000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=100001, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    
    parser.add_argument('--window_length', default=5, type=int, help='')
    parser.add_argument('--combine_state', default= False)
    parser.add_argument('--num_measurements', default=8,type=int)
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO
    parser.add_argument('--num_beams_per_UE',default=8,type=int)
    parser.add_argument('--enable_baseline',default=False)
    parser.add_argument('--enable_genie',default=False)
    parser.add_argument('--ue_speed',default=10)
    parser.add_argument('--full_observation', default=False)
    parser.add_argument('--conv2d_1_kernel_size',type=int,default=5)
    parser.add_argument('--conv2d_2_kernel_size',type=int,default=3)
    parser.add_argument('--oversampling_factor',type=int,default=1)
    parser.add_argument('--num_antennas',type=int,default=64)
    parser.add_argument('--use_saved_traj_in_validation',default=False)
    parser.add_argument('--actor_lambda',type=float,default=0.5)
    
    parser.add_argument('--debug', default = True, dest='debug')

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
    # agent = multiwindow_DDPG(nb_states, nb_actions, args)
    agent = SubAction_DDPG(nb_states, nb_actions, args)
    # evaluate = Evaluator(num_episodes = args.validate_episodes, 
    #                      interval = args.validate_steps, 
    #                      save_path = args.output, 
    #                      max_episode_length=args.max_episode_length,
    #                      use_saved_traj= args.use_saved_traj_in_validation)
    evaluate = BeamSelectionEvaluator(num_episodes = args.validate_episodes, 
                         interval = args.validate_steps, 
                         save_path = args.output, 
                         max_episode_length=args.max_episode_length,
                         use_saved_traj= args.use_saved_traj_in_validation)    
    
    evaluate = None
    if args.mode == 'train':
        tic = time.time()
        rewards = train(args.train_iter, agent, env, evaluate, 
            args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)
        toc = time.time()
        print('Training time for {} steps = {} seconds'.format(args.train_iter, toc-tic))
        plt.figure()
        sns.kdeplot(rewards['agent_rewards'],label='agent')
        if args.enable_baseline:
            sns.kdeplot(rewards['baseline_rewards'],label='baseline')
        if args.enable_genie:
            sns.kdeplot(rewards['genie_rewards'],label='genie')
        plt.legend();
        
        plt.figure()
        plt.plot(rewards['agent_rewards'])
        plt.xlabel('training steps')
        plt.ylabel('avg episode reward')
        
        if evaluate is not None:
            plt.figure()
            plt.plot(rewards['evalulation_rewards'])
            plt.xlabel('training steps')
            plt.ylabel('eval avg episode reward')
        
        if 'critic_mse' in agent.training_log.keys():
            plt.figure()
            plt.plot(agent.training_log['critic_mse'],label='critic MSE')
            plt.legend()
            plt.show()
        if 'actor_mse' in agent.training_log.keys():            
            plt.figure()
            plt.plot(agent.training_log['actor_mse'],label='actor MSE')
            plt.legend()
            plt.show()
        if 'actor_value' in agent.training_log.keys():            
            plt.figure()
            plt.plot(agent.training_log['actor_value'],label='actor Value')
            plt.legend()
            plt.show()
        if 'actor_total' in agent.training_log.keys():                        
            plt.figure()
            plt.plot(agent.training_log['actor_total'],label='actor Total')
            plt.xlabel('training steps')
            plt.legend()
            plt.show()
        
        # if args.enable_baseline and args.enable_genie:
        #     plt.figure()
        #     sns.kdeplot(rewards['agent_rewards'],label='agent')
        #     sns.kdeplot(rewards['baseline_rewards'],label='baseline')
        #     sns.kdeplot(rewards['genie_rewards'],label='genie')
        #     plt.legend();
        #     plt.figure()
        #     plt.plot(rewards['agent_rewards'])
        #     plt.xlabel('episodes')
        #     plt.ylabel('avg episode reward')
        # elif args.enable_baseline:
        #     plt.figure()
        #     sns.kdeplot(rewards['agent_rewards'],label='agent')
        #     sns.kdeplot(rewards['baseline_rewards'],label='baseline')
        #     plt.legend();
        #     plt.figure()
        #     plt.plot(rewards['agent_rewards'])
        #     plt.xlabel('episodes')
        #     plt.ylabel('avg episode reward')
        # elif args.enable_genie:
        #     plt.figure()
        #     sns.kdeplot(rewards[0],label='agent')
        #     sns.kdeplot(rewards[2],label='genie')
        #     plt.legend();
        #     plt.figure()
        #     plt.plot(rewards[0])
        #     plt.xlabel('episodes')
        #     plt.ylabel('avg episode reward')
        # else:
        #     plt.figure()
        #     sns.kdeplot(rewards,label='agent')
        #     plt.legend();
        #     plt.figure()
        #     plt.plot(rewards)
        #     plt.xlabel('episodes')
        #     plt.ylabel('avg episode reward')

        
            
                    

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
