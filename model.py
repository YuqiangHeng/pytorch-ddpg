
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as debug

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
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,kernel_size=8,stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,kernel_size=4,stride=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernal_size=2, stride=2))
        n_size = self._get_conv_output((window_len,nb_states))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_size, hidden1)
        self.fc2 = nn.Linear(hidden1,nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
        
    def _forward_features(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    def _get_conv_output(self,shape):
        batch_size = 1
        dummy_input = Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(dummy_input)
        n_size = output_feat.data.view(batch_size,-1).size(1)
        return n_size
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.conv1(x)
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
        self.fc1 = nn.Linear(int(window_len*nb_states), hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(self.flatten(x))
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out    