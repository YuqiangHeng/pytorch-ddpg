
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from ipdb import set_trace as debug

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

# class Actor(nn.Module):
#     def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(nb_states, hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.fc3 = nn.Linear(hidden2, nb_actions)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         self.init_weights(init_w)
    
#     def init_weights(self, init_w):
#         self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
#         self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
#         self.fc3.weight.data.uniform_(-init_w, init_w)
    
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.relu(out)
#         out = self.fc3(out)
#         out = self.tanh(out)
#         return out


# class Critic(nn.Module):
#     def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(nb_states, hidden1)
#         self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
#         self.fc3 = nn.Linear(hidden2, 1)
#         self.relu = nn.ReLU()
#         self.init_weights(init_w)
    
#     def init_weights(self, init_w):
#         self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
#         self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
#         self.fc3.weight.data.uniform_(-init_w, init_w)
    
#     def forward(self, xs):
#         x, a = xs
#         out = self.fc1(x)
#         out = self.relu(out)
#         # debug()
#         out = self.fc2(torch.cat([out,a],1))
#         out = self.relu(out)
#         out = self.fc3(out)
#         return out
    
#Actor    
class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, window_len, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        # self.conv1 = nn.Sequential(nn.Conv2d(1,16,kernel_size=3,stride=1),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(kernel_size = 2, stride=2))
        # self.conv2 = nn.Sequential(nn.Conv2d(16,32,kernel_size=3,stride=1),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,kernel_size=10,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=16))
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,kernel_size=5,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=32))

        n_size = self._get_conv_output((window_len,nb_states))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_size, hidden1)
        self.fc2 = nn.Linear(hidden1,nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # self.init_weights(init_w)
        
    def _forward_features(self,x):
        out = x.unsqueeze(0)
        out = self.conv1(out)
        out = self.conv2(out)
        return out
    
    def _get_conv_output(self,shape):
        batch_size = 1
        dummy_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(dummy_input)
        n_size = output_feat.data.view(batch_size,-1).size(1)
        return n_size
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        # self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc2.weight.data.uniform_(-init_w, init_w)
        # self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        # print(x.size())
        out = x.unsqueeze(1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.tanh(out)
        return out
    
class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, window_len, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,kernel_size=10,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=16))
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,kernel_size=5,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=32))

        n_size = self._get_conv_output((window_len,nb_states))
        self.fc_state_1 = nn.Linear(n_size, hidden1)
        self.fc_state_2 = nn.Linear(hidden1, hidden2)
        self.fc_action_1 = nn.Linear(nb_actions, hidden1)
        self.fc_action_2 = nn.Linear(hidden1, hidden2)
        self.fc_combined_1 = nn.Linear(hidden2 + hidden2, hidden2)
        self.fc_combined_2 = nn.Linear(hidden2, hidden2)
        self.fc_combined_3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc_state_1.weight.data = fanin_init(self.fc_state_1.weight.data.size())
        self.fc_state_2.weight.data = fanin_init(self.fc_state_2.weight.data.size())
        self.fc_action_1.weight.data = fanin_init(self.fc_action_1.weight.data.size())
        self.fc_action_2.weight.data = fanin_init(self.fc_action_2.weight.data.size())
        self.fc_combined_1.weight.data = fanin_init(self.fc_combined_1.weight.data.size())
        self.fc_combined_2.weight.data = fanin_init(self.fc_combined_2.weight.data.size())
        self.fc_combined_3.weight.data.uniform_(-init_w, init_w)
        
    def _forward_features(self,x):
        out = x.unsqueeze(0)
        out = self.conv1(out)
        out = self.conv2(out)
        return out
    
    def _get_conv_output(self,shape):
        batch_size = 1
        dummy_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(dummy_input)
        n_size = output_feat.data.view(batch_size,-1).size(1)
        return n_size    
    
    def forward(self, xs):
        x, a = xs
        out_state = x.unsqueeze(1)
        out_state = self.conv1(out_state)
        out_state = self.conv2(out_state)
        out_state = self.flatten(out_state)
        out_state = self.fc_state_1(out_state)
        out_state = self.relu(out_state)
        out_state = self.fc_state_2(out_state)
        out_state = self.relu(out_state)
        out_action = self.fc_action_1(a)
        out_action = self.relu(out_action)
        out_action = self.fc_action_2(out_action)
        out_action = self.relu(out_action)
        # print(out_state.shape, out_action.shape)
        out = torch.cat((out_state, out_action), 1)
        out = self.fc_combined_1(out)
        out = self.relu(out)
        out = self.fc_combined_2(out)
        out = self.relu(out)
        out = self.fc_combined_3(out)
        out = self.tanh(out)
        return out    
    
 
# from BeamManagement import BeamManagementEnv, BeamManagementEnvMultiFrame
# if __name__ == "__main__":
#     train_iter = int(1e5)
    
#     window_length = 5
#     env = BeamManagementEnv(ue_speed = 15)
#     nb_states = env.observation_space.shape[0]
#     nb_actions = env.action_space.shape[0]
#     agent = DDPG(nb_states, nb_actions, window_len, args)
#     train(args.train_iter, agent, env, evaluate, args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)
    
    
    
    
    
    
    
    