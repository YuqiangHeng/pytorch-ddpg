
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic,MLP,SerializedCritic)
from memory import BeamSpaceSequentialMemory
from util import *

import argparse
from copy import deepcopy

from evaluator import Evaluator, BeamSelectionEvaluator, DDPGAgentEval
from BeamManagementEnv import BeamManagementEnv
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import seaborn as sns


# from ipdb import set_trace as debug

criterion = nn.MSELoss()

class BeamPredictor(object):
    def __init__(self, nb_states, nb_actions, args):
        self.seed = args.seed
        if self.seed > 0:
            self.seed(self.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        self.num_beams_per_UE = args.num_beams_per_UE
        self.num_measurements = args.num_measurements
        self.window_length = args.window_length
        
        self.actor = MLP(self.nb_states, self.window_length, self.num_measurements)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)
        self.memory_limit = args.rmsize
        #Create replay buffer
        # self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.memory = BeamSpaceSequentialMemory(limit=self.memory_limit, window_length=self.window_length, num_measurements=self.num_measurements)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.ob_t = None # Most recent observation
        self.a_t = None # Most recent action
        self.is_training = True
        self.debug = args.debug
        self.training_log = {'prediction_masked_mse':[],'prediction_all_mse':[]}

        # 
        if USE_CUDA: self.cuda()

    def update_predictor(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        next_observation_batch = next_state_batch[:,-self.num_measurements:,:]

        # Actor update
        self.actor.zero_grad()
        
        # Beam qual prediction loss, only if using MSE of actor output
        current_state_mask = np.zeros((self.batch_size,self.num_measurements*self.window_length,self.nb_states))
        for i in range(self.batch_size):
            for j in range(self.num_measurements*self.window_length):
                current_state_mask[i,j,np.argsort(state_batch[i,j,:])[-self.num_beams_per_UE:]] = 1
                
        next_observation_mask = np.zeros((self.batch_size,self.num_measurements,self.nb_states))
        for i in range(self.batch_size):
            for j in range(self.num_measurements):
                next_observation_mask[i,j,np.argsort(next_observation_batch[i,j,:])[-self.num_beams_per_UE:]] = 1  
                
        states = to_tensor(state_batch)
        states_masked = torch.mul(states, to_tensor(current_state_mask))
        predicted_beam_qual = self.actor(states_masked)
        predicted_beam_qual_masked = torch.mul(predicted_beam_qual, to_tensor(next_observation_mask))
        true_beam_qual = to_tensor(next_observation_batch)
        true_beam_qual_masked = torch.mul(true_beam_qual, to_tensor(next_observation_mask))
        prediction_masked_mse = criterion(predicted_beam_qual_masked,true_beam_qual_masked)    
        prediction_all_mse = criterion(predicted_beam_qual,true_beam_qual)    
        self.training_log['prediction_masked_mse'].append(prediction_masked_mse.item())
        self.training_log['prediction_all_mse'].append(prediction_all_mse.item())
        
        prediction_masked_mse.backward()
        self.actor_optim.step()
        
        return prediction_masked_mse.item(), prediction_all_mse.item()
 
       
    def predict(self, current_observation, next_observation):
        current_state = self.memory.get_recent_state(current_observation)
        current_state = np.array([np.squeeze(np.array(current_state))]) #add batch dim
        
        current_state_mask = np.zeros(current_state.shape)
        for j in range(self.num_measurements*self.window_length):
            current_state_mask[0,j,np.argsort(current_state[0,j,:])[-self.num_beams_per_UE:]] = 1
        
        next_observation = np.array([np.squeeze(np.array(next_observation))]) #add batch dim
        next_observation_mask = np.zeros(next_observation.shape)
        for j in range(self.num_measurements):
            next_observation_mask[0,j,np.argsort(next_observation[0,j,:])[-self.num_beams_per_UE:]] = 1  
        
        current_state_masked = torch.mul(to_tensor(current_state), to_tensor(current_state_mask))
        predicted_beam_qual = self.actor(current_state_masked)
        predicted_beam_qual_masked = torch.mul(predicted_beam_qual, to_tensor(next_observation_mask))
        true_beam_qual = to_tensor(next_observation)
        true_beam_qual_masked = torch.mul(true_beam_qual, to_tensor(next_observation_mask))
        masked_mse = criterion(predicted_beam_qual_masked, true_beam_qual_masked)
        all_mse = criterion(predicted_beam_qual, true_beam_qual)
        return masked_mse.item(), all_mse.item()
    
    def get_eval_predictor(self):
        eval_predictor = deepcopy(self)
        eval_predictor.memory = BeamSpaceSequentialMemory(limit=self.memory_limit, window_length=self.window_length, num_measurements=self.num_measurements)
        return eval_predictor

    def eval(self):
        self.actor.eval()


    def cuda(self):
        self.actor.cuda()


    def observe(self, r_t, ob_t1, done):
        if self.is_training:
            self.memory.append(self.ob_t, self.a_t, r_t, done)
            self.ob_t = ob_t1
     
    # modified reset() function that also takes in initial beam config
    def reset(self, obs):
        self.ob_t = obs
        # self.a_t = beams

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )


    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)

class BeamPredictionEvaluator(object):

    def __init__(self, num_episodes, interval, save_path='', max_episode_length=None, use_saved_traj=False):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes,0)
        self.use_saved_traj = use_saved_traj

    def __call__(self, env:BeamManagementEnv, agent:BeamPredictor, debug=False, visualize=False, save=False):
        masked_mse = []
        all_mse = []
        
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
                observation = deepcopy(env.reset())
                agent.reset(observation)
                episode_steps = 0
                episode_masked_mse = 0.
                episode_all_mse = 0.
            # start episode
            action = np.zeros(env.codebook_size)
            action[0:env.num_beams_per_UE] = 1
            observation2, reward, done, info = env.step(action)
            observation2 = deepcopy(observation2)
            if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                done = True
            
            masked_mse_temp, all_mse_temp = agent.predict(observation, observation2)
            episode_masked_mse += masked_mse_temp
            episode_all_mse += all_mse_temp
            
            agent.observe(reward, observation2, done)
        
            #update
            step += 1
            episode_steps += 1
            observation = deepcopy(observation2)

            if done: # end of episode
                masked_mse.append(episode_masked_mse/episode_steps)
                all_mse.append(episode_all_mse/episode_steps)
                
                if debug:
                    print('Episode#{} Masked MSE={} Full MSE={}'.format(episode,episode_masked_mse/episode_steps,episode_all_mse/episode_steps))
    
                agent.memory.append(
                    observation,
                    action,
                    0., False
                )
                # reset
                observation = None
                episode_steps = 0
                episode_reward = 0.
                episode += 1
        
        # allow env to generate traj
        env.set_data_mode(use_saved_trajectory = False, num_saved_traj = None)
                
        
        return masked_mse, all_mse


        
def train(num_iterations, agent:BeamPredictor, env:BeamManagementEnv,  evaluate:BeamPredictionEvaluator, validate_steps, output, max_episode_length=None, debug=False):
    
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    train_episode_masked_mse = 0
    train_episode_full_mse = 0
    train_masked_mse = []
    train_full_mse = []
    train_times = []
    if evaluate is not None:
        eval_masked_mse = []
        eval_full_mse = []
        eval_times = []
        
    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        action = np.zeros(env.codebook_size)
        action[0:env.num_beams_per_UE] = 1
        
        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup :
            t_masked_mse, t_full_mse = agent.update_predictor()
            train_episode_masked_mse += t_masked_mse
            train_episode_full_mse += t_full_mse
        
        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            # policy = lambda x: agent.select_action(x, decay_epsilon=False)
            tic = time.time()
            evalagent = agent.get_eval_predictor()
            val_env = deepcopy(env)
            val_env.reset()
            masked_mse, all_mse = evaluate(val_env, evalagent, debug=False, visualize=False, save = False)
            toc = time.time()
            # if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
            print('[Evaluate] Step_{:07d}: time:{} seconds, avg masked mse:{}, avg full mse:{}'.format(step, toc-tic, np.mean(masked_mse), np.mean(all_mse)))
            eval_masked_mse.append(np.mean(masked_mse))
            eval_full_mse.append(np.mean(all_mse))
            eval_times.append(step)
            
            plt.figure()
            sns.kdeplot(masked_mse,label='masked MSE')
            sns.kdeplot(all_mse,label='full MSE')
            plt.legend();
            plt.title('Eval Results after #{} Training Steps'.format(step))
            plt.show()


        # update 
        step += 1
        episode_steps += 1
        observation = deepcopy(observation2)

        if done: # end of episode
            # if debug: prGreen('#{}: mean_episode_reward:{} episode_reward:{} steps:{}'.format(episode,episode_reward/episode_steps,episode_reward,step))
            agent.memory.append(
                observation,
                action,
                0., False
            )
            
            train_masked_mse.append(train_episode_masked_mse/episode_steps)
            train_full_mse.append(train_episode_full_mse/episode_steps)
            train_times.append(step)
            # reset
            observation = None
            train_episode_masked_mse = 0
            train_episode_full_mse = 0
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            
    all_results = {'eval_masked_mse':eval_masked_mse,'eval_full_mse':eval_full_mse,'eval_times':eval_times,'train_masked_mse':train_masked_mse,'train_full_mse':train_full_mse,'train_times':train_times}
    return all_results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='beam', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.5, type=float, help='')
    parser.add_argument('--bsize', default=128, type=int, help='minibatch size')
    # parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--rmsize', default=60000, type=int, help='memory size')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=100, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=5000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=200001, type=int, help='train iters each timestep')
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
    # agent = multiwindow_DDPG(nb_states, nb_actions, args)
    agent = BeamPredictor(nb_states, nb_actions, args)
    # evaluate = Evaluator(num_episodes = args.validate_episodes, 
    #                      interval = args.validate_steps, 
    #                      save_path = args.output, 
    #                      max_episode_length=args.max_episode_length,
    #                      use_saved_traj= args.use_saved_traj_in_validation)
    evaluate = BeamPredictionEvaluator(num_episodes = args.validate_episodes, 
                         interval = args.validate_steps, 
                         save_path = args.output, 
                         max_episode_length=args.max_episode_length,
                         use_saved_traj= args.use_saved_traj_in_validation)    
    # evaluate = None
    if args.mode == 'train':
        tic = time.time()
        rewards = train(args.train_iter, agent, env, evaluate, 
            args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)
        toc = time.time()
        print('Training time for {} steps = {} seconds'.format(args.train_iter, toc-tic))
        
        plt.figure()
        plt.plot(rewards['train_times'],rewards['train_masked_mse'], label = 'train masked mse')
        plt.plot(rewards['train_times'],rewards['train_masked_mse'], label = 'train masked mse')
        plt.legend();
        plt.figure()
        plt.plot(rewards['eval_times'],rewards['eval_masked_mse'], label = 'train masked mse')
        plt.plot(rewards['eval_times'],rewards['eval_masked_mse'], label = 'train masked mse')
        plt.legend();
        
        # plt.figure()
        # sns.kdeplot(rewards['agent_rewards'],label='agent')
        # if args.enable_baseline:
        #     sns.kdeplot(rewards['baseline_rewards'],label='baseline')
        # if args.enable_genie:
        #     sns.kdeplot(rewards['genie_rewards'],label='genie')
        # plt.legend();
        
        # plt.figure()
        # plt.plot(rewards['agent_rewards'])
        # plt.xlabel('training steps')
        # plt.ylabel('avg episode reward')
        
        # if evaluate is not None:
        #     plt.figure()
        #     plt.plot(rewards['evalulation_rewards'])
        #     plt.xlabel('training steps')
        #     plt.ylabel('eval avg episode reward')
        
        # plt.figure()
        # plt.plot(agent.training_log['critic_mse'],label='critic MSE')
        # plt.legend()
        # plt.show()
        # plt.figure()
        # plt.plot(agent.training_log['actor_mse'],label='actor MSE')
        # plt.legend()
        # plt.show()
        # plt.figure()
        # plt.plot(agent.training_log['actor_value'],label='actor Value')
        # plt.legend()
        # plt.show()
        # plt.figure()
        # plt.plot(agent.training_log['actor_total'],label='actor Total')
        # plt.xlabel('training steps')
        # plt.legend()
        # plt.show()
        
        

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
