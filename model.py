
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

def conv_output_size(shape,padding,dilation,kernel_size,stride):
    return np.floor((shape + 2 * padding - dilation * (kernel_size - 1)-1) / stride + 1)

def conv2d_output_dim(shape,padding,dilation,kernel_size,stride):
    return (conv_output_size(shape[0],padding[0],dilation[0],kernel_size[0],stride[0]),conv_output_size(shape[1],padding[1],dilation[1],kernel_size[1],stride[1]))


    

        
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
    def __init__(self, nb_states, nb_actions, window_len, hidden1=400, hidden2=300, kernel_size_1=3,kernel_size_2 = 3, init_w=3e-3):
        super(Actor, self).__init__()
        # self.conv1 = nn.Sequential(nn.Conv2d(1,16,kernel_size=3,stride=1),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(kernel_size = 2, stride=2))
        # self.conv2 = nn.Sequential(nn.Conv2d(16,32,kernel_size=3,stride=1),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,kernel_size=kernel_size_1,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=16))
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,kernel_size=kernel_size_2,stride=1),
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
        
    def select_beams(self, x, nb_actions, n):
        bsize = x.shape[0]
        binary_beams = np.zeros((bsize, nb_actions))
        for i in range(bsize):
            sel = np.argsort(x[i])[-n:]
            binary_beams[i,sel] = 1
        return binary_beams
    
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
    def __init__(self, nb_states, nb_actions, window_len, hidden1=400, hidden2=300, kernel_size_1=3,kernel_size_2 = 3, init_w=3e-3):
        super(Critic, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,kernel_size=kernel_size_1,stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=16))
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,kernel_size=kernel_size_2,stride=1),
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

class SerializedCritic(nn.Module):
    def __init__(self, nb_states, nb_actions, window_len, num_measurements):
        super().__init__()
        self.flatten = nn.Flatten()
        self.state_input_shape = (window_len*num_measurements, nb_states)
        s_nsize = self._serial_input_size(self.state_input_shape)
        # s_hidden1 = int(s_nsize/2)
        # s_hidden2 = int(s_nsize/4)
        # s_hidden3 = int(s_nsize/8)
        # self.action_input_shape = (nb_actions)
        # a_nsize = self._serial_input_size(self.action_input_shape)
        self.action_input_shape = nb_actions
        a_nsize = nb_actions
        # a_hidden1 = int(a_nsize/2)
        # a_hidden2 = int(a_nsize/4)
        # a_hidden3 = int(a_nsize/8)
        
        comb_input_size = s_nsize + a_nsize
        comb_hidden1 = int(comb_input_size/2)
        comb_hidden2 = int(comb_input_size/4)
        comb_hidden3 = int(comb_input_size/8)
        comb_hidden4 = int(comb_input_size/16)
        
        # self.s_fc1 = nn.Sequential(nn.Linear(s_nsize,s_hidden1),
        #                             nn.ReLU())
        # self.s_fc2 = nn.Sequential(nn.Linear(s_hidden1,s_hidden2),
        #                             nn.ReLU())
        # self.s_fc3 = nn.Sequential(nn.Linear(s_hidden2,s_hidden2),
        #                             nn.ReLU())        
        # self.a_fc1 = nn.Sequential(nn.Linear(a_nsize,a_hidden1),
        #                             nn.ReLU())
        # self.a_fc2 = nn.Sequential(nn.Linear(a_hidden1,a_hidden2),
        #                             nn.ReLU())        
        # self.a_fc3 = nn.Sequential(nn.Linear(a_hidden2,a_hidden3),
        #                             nn.ReLU())   
        self.comb_fc1 = nn.Sequential(nn.Linear(comb_input_size,comb_hidden1),
                                    nn.ReLU())   
        self.comb_fc2 = nn.Sequential(nn.Linear(comb_hidden1,comb_hidden2),
                                    nn.ReLU())   
        self.comb_fc3 = nn.Sequential(nn.Linear(comb_hidden2,comb_hidden3),
                                    nn.ReLU())   
        self.comb_fc4 = nn.Sequential(nn.Linear(comb_hidden3,comb_hidden4),
                                    nn.ReLU())   
        self.comb_fc5 = nn.Linear(comb_hidden4,1)        
        
    def _serial_input_size(self, shape):
        batch_size = 1
        dummy_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        dummy_flat = self.flatten(dummy_input)
        nsize = dummy_flat.data.view(batch_size,-1).size(1)
        return nsize
    
    def forward(self, xs):
        s, a = xs
        state_flattened = self.flatten(s)
        action_flattened = self.flatten(a)
        combined = torch.cat((state_flattened,action_flattened),1)
        out = self.comb_fc1(combined)
        out = self.comb_fc2(out)
        out = self.comb_fc3(out)
        out = self.comb_fc4(out)
        out = self.comb_fc5(out)
        return out    
        
    
class MLP(nn.Module):
    def __init__(self, nb_states, window_len, num_measurements):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_shape = (window_len*num_measurements, nb_states)
        nsize = self._serial_input_size(self.input_shape)
        hidden1 = int(nsize/2)
        hidden2 = int(nsize/4)
        hidden3 = int(nsize/8)
        self.outdim = num_measurements*nb_states
        self.outshape = (num_measurements,nb_states)
        self.fc1 = nn.Sequential(nn.Linear(nsize,hidden1),
                                    nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden1,hidden2),
                                    nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(hidden2,hidden2),
                                    nn.ReLU())
        # self.fc4 = nn.Sequential(nn.Linear(hidden2,hidden2),
        #                             nn.ReLU())
        self.fc5 = nn.Sequential(nn.Linear(hidden2,self.outdim),
                                    nn.ReLU())
        self.fc6 = nn.Linear(self.outdim,self.outdim)

    def _serial_input_size(self, shape):
        batch_size = 1
        dummy_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        dummy_flat = self.flatten(dummy_input)
        nsize = dummy_flat.data.view(batch_size,-1).size(1)
        return nsize
    
    def select_beams(self, x, nb_actions, n):
        bsize = x.shape[0]
        binary_beams = np.zeros((bsize, nb_actions))
        for i in range(bsize):
            sel = []
            pool = list(np.arange(nb_actions))
            while len(sel) < n:
                temp_r = [x[i,:,np.array(sel+[b])].max(axis=1).sum() for b in pool]
                temp_sel = pool[np.argmax(temp_r)]
                sel.append(temp_sel)
                pool.remove(temp_sel)
            # sel = np.argsort(np.sum(x[i],axis=0))[-n:]
            binary_beams[i,sel] = 1
        return binary_beams

    def forward(self,x):
        flat = self.flatten(x)
        out = self.fc1(flat)
        out = self.fc2(out)
        out = self.fc3(out)
        # out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = out.view(-1,*self.outshape)
        return out

class MLPDist(nn.Module):
    def __init__(self, nb_states, window_len, num_measurements):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_shape = (window_len*num_measurements, nb_states)
        nsize = self._serial_input_size(self.input_shape)
        hidden1 = int(nsize/2)
        hidden2 = int(nsize/4)
        hidden3 = int(nsize/8)
        self.outdim = nb_states
        self.outshape = nb_states
        self.fc1 = nn.Sequential(nn.Linear(nsize,hidden1),
                                    nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden1,hidden2),
                                    nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(hidden2,hidden2),
                                    nn.ReLU())
        # self.fc4 = nn.Sequential(nn.Linear(hidden2,hidden2),
        #                             nn.ReLU())
        self.fc5 = nn.Sequential(nn.Linear(hidden2,self.outdim),
                                    nn.ReLU())
        self.fc6 = nn.Sequential(nn.Linear(self.outdim,self.outdim),
                                 nn.Tanh())

    def _serial_input_size(self, shape):
        batch_size = 1
        dummy_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        dummy_flat = self.flatten(dummy_input)
        nsize = dummy_flat.data.view(batch_size,-1).size(1)
        return nsize
    
    def select_beams(self, x, nb_actions, n):
        bsize = x.shape[0]
        binary_beams = np.zeros((bsize, nb_actions))
        for i in range(bsize):
            sel = np.argsort(x[i])[-n:]
            binary_beams[i,sel] = 1
        return binary_beams
    
    def forward(self,x):
        flat = self.flatten(x)
        out = self.fc1(flat)
        out = self.fc2(out)
        out = self.fc3(out)
        # out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        return out

class SerializedAutonEncoder(nn.Module):
    def __init__(self, nb_states, window_len, num_measurements):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_shape = (window_len*num_measurements, nb_states)
        nsize = self._serial_input_size(self.input_shape)
        hidden1 = int(nsize/2)
        hidden2 = int(nsize/4)
        hidden3 = int(nsize/8)
        self.outdim = num_measurements*nb_states
        self.outshape = (num_measurements,nb_states)
        self.encode1 = nn.Sequential(nn.Linear(nsize,hidden1),
                                    nn.ReLU())
        self.encode2 = nn.Sequential(nn.Linear(hidden1,hidden2),
                                    nn.ReLU())
        self.encode3 = nn.Sequential(nn.Linear(hidden2,hidden3),
                                    nn.ReLU())
        self.decode1 = nn.Sequential(nn.Linear(hidden3,hidden2),
                                    nn.ReLU())
        self.decode2 = nn.Sequential(nn.Linear(hidden2,self.outdim),
                                    nn.ReLU())
        self.decode3 = nn.Linear(self.outdim,self.outdim)

    def _serial_input_size(self, shape):
        batch_size = 1
        dummy_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        dummy_flat = self.flatten(dummy_input)
        nsize = dummy_flat.data.view(batch_size,-1).size(1)
        return nsize

    def select_beams(self, x, nb_actions, n):
        bsize = x.shape[0]
        binary_beams = np.zeros((bsize, nb_actions))
        for i in range(bsize):
            sel = np.argsort(np.sum(x[i],axis=0))[-n:]
            binary_beams[i,sel] = 1
        return binary_beams

    def forward(self,x):
        flat = self.flatten(x)
        enc = self.encode1(flat)
        enc = self.encode2(enc)
        enc = self.encode3(enc)
        dec = self.decode1(enc)
        dec = self.decode2(dec)
        dec = self.decode3(dec)
        out = dec.view(-1,*self.outshape)
        return out
    
class ConvAutoEncoder(nn.Module):
    def __init__(self, nb_states, nb_actions, window_len):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1),
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(2,2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,4,kernel_size=3,stride=(2,1),padding=1),
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(2,2))
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(4,16,kernel_size=(2,2),stride=(1,2),padding=(0,0)),
                                    nn.LeakyReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(16,1,kernel_size=(2,2),stride=(2,2),padding=(1,0)),
                                    nn.LeakyReLU())
        self.input_shape = (window_len,nb_states)
        
        self._print_encoded_shape()
        self._print_decoded_shape()
        
    def _encode_features(self,x):
        out = x.unsqueeze(0)
        out = self.conv1(out)
        out = self.conv2(out)
        return out
    
    def _print_encoded_shape(self):
        batch_size = 1
        dummy_input = torch.autograd.Variable(torch.rand(batch_size, *self.input_shape))
        output_feat = self._encode_features(dummy_input)
        print('latent dimension:',output_feat.shape)
    
    def _print_decoded_shape(self):
        batch_size = 1
        dummy_input = torch.autograd.Variable(torch.rand(batch_size, *self.input_shape))
        output_feat = self.forward(dummy_input)
        print('decoded dimension:',output_feat.shape)

    def select_beams(self, x, nb_actions, n):
        bsize = x.shape[0]
        binary_beams = np.zeros((bsize, nb_actions))
        for i in range(bsize):
            sel = np.argsort(np.sum(x[i],axis=0))[-n:]
            binary_beams[i,sel] = 1
        return binary_beams
    
    def forward(self, x):
        # print(x.size())
        encoded = x.unsqueeze(1)
        encoded = self.conv1(encoded)
        encoded = self.conv2(encoded)
        decoded = self.deconv1(encoded)
        decoded = self.deconv2(decoded)
        return decoded.squeeze(1)
        
class SubActionNN(nn.Module):
    def __init__(self, nb_states):
        self.inputdim = nb_states
        self.hidden1_dim = int(nb_states/2)
        self.hidden2_dim = int(nb_states/4)
        self.fc1 = nn.Sequential(nn.Linear(self.inputdim,self.hidden1_dim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(self.hidden1_dim,self.hidden2_dim), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(self.hidden2_dim,1), nn.Sigmoid())
    
    def forward(self,x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
class SubActionActor(nn.Module):
    def __init__(self, nb_states，nb_actions, window_len, num_measurements):
        super().__init__()
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.window_len = window_len
        self.num_measurements = num_measurements
        self.input_shape = (self.window_len * self.num_measurements, nb_states)
        self.lstm = nn.LSTM(input_size = self.nb_states, hidden_size = self.nb_states, num_layers = 2, batch_first=True, dropout = 0.2)
        self.fc_out_dim = int(self.nb_states/2)
        self.fc = nn.Sequential(nn.Linear(self.nb_states,self.fc_out_dim), nn.ReLU())
        self.sub_action_nns = [SubActionNN(self.nb_states) for i in range(self.nb_actions)]
                
    def forward(self, x):
        out, hidden = self.lstm(x)
        sub_action_out = []
        for nn in self.sub_action_nns:
            sub_action_score = nn(out)
            sub_action_out.append(sub_action_out)
        return sub_action_out
    
    def select_beams(self, x, nb_actions, n):
        bsize = x.shape[0]
        binary_beams = np.zeros((bsize, nb_actions))
        for i in range(bsize):
            sel = np.argsort(x[i])[-n:]
            binary_beams[i,sel] = 1
        return binary_beams
        
class SubActionCritic(nn.Module):
    def __init__(self, nb_states，nb_actions, window_len, num_measurements):
        super().__init__()
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.window_len = window_len
        self.num_measurements = num_measurements
        self.input_shape = (self.window_len * self.num_measurements, nb_states)
        self.lstm = nn.LSTM(input_size = self.nb_states, hidden_size = self.nb_states, num_layers = 2, batch_first=True, dropout = 0.2)
        self.fc_out_dim = int(self.nb_states/2)
        self.fc = nn.Sequential(nn.Linear(self.nb_states,self.fc_out_dim), nn.ReLU())
        self.sub_action_nns = [SubActionNN(self.nb_states) for i in range(self.nb_actions)]
        
    def forward(self, x):
        s,a = x
        lstm_out, lstm_hidden = self.lstm(s)
        for i in np.nonzeros(a)
        
        
        
        

        
from BeamManagementEnv import BeamManagementEnv
from memory import SequentialMemory,BeamSpaceSequentialMemory,RingBuffer
from torch.autograd import Variable
from collections import deque
from tqdm import tqdm
import pickle

# FLOAT = torch.FloatTensor
# def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
#     return Variable(
#         torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
#     ).type(dtype)


# if __name__ == "__main__":
#     window_len = 4
#     ue_speed = 15
#     num_measurements = 16
#     num_ob = int(1e6)
#     q = deque(maxlen=window_len+1)
#     env = BeamManagementEnv(ue_speed=ue_speed,num_beams_per_UE=64,num_measurements=num_measurements,min_traj_len = 10,full_observation=True)
#     nb_actions = env.action_space.shape[0]
#     nb_states = env.observation_space.shape[0]
#     a_t = np.ones((nb_actions))
#     train_episode_idx = 0
#     train_step_idx = 0
#     train_loss = 0
#     done = False  
#     ob_idx = 0
#     ob_t = env.reset()
#     q.append(ob_t)
#     x = []
#     y = []
#     pbar = tqdm(total=num_ob)
#     while ob_idx < num_ob:
#         ob_t_1, r_t, done, info = env.step(a_t)
#         q.append(ob_t_1)
#         if len(q) == q.maxlen:
#             x.append(np.array(q)[0:-1,:,:].reshape(window_len*num_measurements,-1))
#             y.append(np.array(q[-1]))
#             ob_idx += 1
#             pbar.update(1)
#         if done:
#             ob_t = env.reset()
#             q.clear()
#             q.append(ob_t)
#     x = np.array(x)
#     y = np.array(y)
#     print(x.shape,y.shape)
#     np.save('x',x)
#     np.save('y',y)
#     pbar.close()


# from torchvision import datasets
# from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

class CustomDataset():
    def __init__(self, xfname, yfname, maxlen):
        self.x = np.load(xfname)
        self.y = np.load(yfname)
        if len(self.y) > maxlen:
            self.x = self.x[0:maxlen]
            self.y = self.y[0:maxlen]

    def __getitem__(self, index):
        # This method should return only 1 sample and label 
        # (according to "index"), not the whole dataset
        # So probably something like this for you:
        return self.x[index,:,:], self.y[index,:]

    def __len__(self):
        return len(self.y)
    
# if __name__ == "__main__":
#     window_len = 4
#     ue_speed = 15
#     num_measurements = 16
#     num_sample = int(1e5)
#     num_epochs = int(1e5)
#     # model = ConvAutoEncoder(64,64,window_len*num_measurements).double()
#     model = MLP(64,window_len,num_measurements).double()
#     model.cuda()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     criterion = torch.nn.MSELoss()
#     # dataset = CustomDataset('x.npy','y.npy',num_sample)
#     dataset = pickle.load(open('dataset.pkl','rb'))
#     batch_size = 128
#     validation_split = 0.2
#     shuffle_dataset = True
#     random_seed = 42
#     dataset_size = len(dataset)
#     indices = list(range(dataset_size))
#     split = int(np.floor(validation_split*dataset_size))
#     if shuffle_dataset:
#         np.random.seed(random_seed)
#         np.random.shuffle(indices)
#     train_indices, val_indices = indices[split:], indices[:split]
    
#     train_sampler = SubsetRandomSampler(train_indices)
#     valid_sampler = SubsetRandomSampler(val_indices)
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
#     validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)    
    
#     for epoch in range(num_epochs):
#         avg_loss = 0
#         for batch_index, (x, y) in enumerate(train_loader):
#             x = x.to('cuda', non_blocking=True)
#             y = y.to('cuda', non_blocking=True)
#             predicted_y = model(x)
#             batch_loss = criterion(predicted_y,y)
#             optimizer.zero_grad()
#             batch_loss.backward()
#             optimizer.step()
#             avg_loss += batch_loss.item()
#         # if epoch % 10 == 0: print('Epoch:{} Training Loss:{}'.format(epoch,avg_loss/(len(train_loader))))
#         # if epoch % 10 == 0:
#         #     avg_val_loss = 0
#         #     for batch_index, (x, y) in enumerate(validation_loader):
#         #         x = x.to('cuda', non_blocking=True)
#         #         y = y.to('cuda', non_blocking=True)
#         #         predicted_y = model(x)
#         #         batch_loss = criterion(predicted_y,y)
#         #         avg_val_loss += batch_loss.item()
#         #     print('Epoch:{} Validation Loss:{}'.format(epoch,avg_val_loss/len(validation_loader)))
#         if epoch % 10 == 0:
#             avg_val_loss = 0
#             for batch_index, (x, y) in enumerate(validation_loader):
#                 x = x.to('cuda', non_blocking=True)
#                 y = y.to('cuda', non_blocking=True)
#                 predicted_y = model(x)
#                 batch_loss = criterion(predicted_y,y)
#                 avg_val_loss += batch_loss.item()
#             print('Epoch:{}, Training Loss:{}, Validation Loss:{}'.format(epoch,avg_loss/len(train_loader),avg_val_loss/len(validation_loader)))    


    
    
    # for train_idx in range(train_iter):
        
    # pbar = tqdm(total=num_ob)
    # while ob_idx < num_ob:
    #     ob_t_1, r_t, done, info = env.step(a_t)
    #     q.append(ob_t_1)
    #     if len(q) == q.maxlen:
    #         x.append(np.array(q)[0:-1,:,:].reshape(window_len*num_measurements,-1))
    #         y.append(np.array(q[-1]))
    #         ob_idx += 1
    #         pbar.update(1)
    #     if done:
    #         ob_t = env.reset()
    #         q.clear()
    #         q.append(ob_t)
    # x = np.array(x)
    # y = np.array(y)
    # print(x.shape,y.shape)
    # np.save('x',x)
    # np.save('y',y)
    # pbar.close()    
    
# if __name__ == "__main__":
#     window_len = 4
#     ue_speed = 15
#     num_measurements = 16
#     train_iter = int(1e5)
#     warm_up = 100
#     batch_size = 32
#     ringbuffer = RingBuffer(maxlen = window_len)
#     memory = BeamSpaceSequentialMemory(limit=600000, window_length=window_len, num_measurements=num_measurements)
#     env = BeamManagementEnv(ue_speed=ue_speed,num_beams_per_UE=64,num_measurements=num_measurements,min_traj_len = 10)
#     ob_t = env.reset()
#     nb_actions = env.action_space.shape[0]
#     nb_states = env.observation_space.shape[0]
#     model = ConvAutoEncoder(nb_states=nb_states,nb_actions=nb_actions,window_len=window_len*num_measurements)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     a_t = np.ones((nb_actions))
#     train_episode_idx = 0
#     train_step_idx = 0
#     train_loss = 0
#     done = False
#     while train_episode_idx < train_iter:
#         ob_t_1, r_t, done, info = env.step(a_t)
#         memory.append(ob_t,a_t,r_t,done)
#         ob_t = ob_t_1
#         if train_step_idx > warm_up:
#             state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = memory.sample_and_split(batch_size)
#             predicted_next_state_batch = model(to_tensor(state_batch))
#             loss = criterion(predicted_next_state_batch, to_tensor(next_state_batch))
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#         if done:
#             train_episode_idx += 1
#             if train_step_idx > warm_up:print('Epoch: {} \tTraining Loss: {:.6f}'.format(train_episode_idx, train_loss))
#             train_loss = 0
#             ob_t = env.reset()
#         train_step_idx += 1

        
 
# from BeamManagement import BeamManagementEnv, BeamManagementEnvMultiFrame
# if __name__ == "__main__":
#     train_iter = int(1e5)
    
#     window_length = 5
#     env = BeamManagementEnv(ue_speed = 15)
#     nb_states = env.observation_space.shape[0]
#     nb_actions = env.action_space.shape[0]
#     agent = DDPG(nb_states, nb_actions, window_len, args)
#     train(args.train_iter, agent, env, evaluate, args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)
    
    
    
    
    
    
    
    