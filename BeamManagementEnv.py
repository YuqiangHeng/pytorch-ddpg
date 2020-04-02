# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:16:56 2020

@author: ethan

TODO: change boundary points method to find all points with fewer than 8 neighbors: DONE
TODO: quantize SNR feedback in beam report (state-space representation)
TODO: implement genie(optimal) baseline 
TODO: implement spatial moving window baseline
"""
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
#import matplotlib.pyplot as plt
#from Utils import alpha_shape

h_imag_fname = "H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_imag.npy"
h_real_fname = "H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_real.npy"
ue_loc_fname = "H_Matrices FineGrid/MISO_Static_FineGrid_UE_location.npy"

# all_loc = np.load(ue_loc_fname)[:,0:2]
#boundary_points = [i for i in range(all_loc.shape[0]) if len(np.nonzero(np.linalg.norm(all_loc-all_loc[i],axis=1)<0.6)[0])<9]
##edges = alpha_shape(all_loc[:,0:2], alpha=0.25, only_outer=True)
##boundary_points = np.unique(np.array(list(edges)).flatten())
###plt.scatter(all_loc[boundary_points,0],all_loc[boundary_points,1])
#neighbor_points = np.zeros((all_loc.shape[0],8)) #indices start at 1, remember to -1 when indexing
#"""
#[0][1][2]
#[7][x][3]
#[6][5][4]
#"""
#for i in range(all_loc.shape[0]):
#    dist = np.linalg.norm(all_loc - all_loc[i,:],axis=1)
#    neighbor_idc = np.nonzero(dist<0.6)[0]
#    for j in neighbor_idc:
#        if all_loc[j,0] < all_loc[i,0]:
#            if all_loc[j,1] < all_loc[i,1]:
#                neighbor_points[i,6] = j+1
#            elif all_loc[j,1] == all_loc[i,1]:
#                neighbor_points[i,7] = j+1
#            if all_loc[j,1] > all_loc[i,1]:
#                neighbor_points[i,0] = j+1
#        if all_loc[j,0] == all_loc[i,0]:
#            if all_loc[j,1] < all_loc[i,1]:
#                neighbor_points[i,5] = j+1
#            if all_loc[j,1] > all_loc[i,1]:  
#                neighbor_points[i,1] = j+1
#        if all_loc[j,0] > all_loc[i,0]:
#            if all_loc[j,1] < all_loc[i,1]:
#                neighbor_points[i,4] = j+1
#            elif all_loc[j,1] == all_loc[i,1]:
#                neighbor_points[i,3] = j+1
#            if all_loc[j,1] > all_loc[i,1]:    
#                neighbor_points[i,2] = j+1
#neighbor_dir_lists = [np.nonzero(i)[0] for i in neighbor_points]
#boundary_neighbor_dir_lists = [neighbor_dir_lists[i] for i in boundary_points]   
#boundary_neighbor_mid_dir = []       
#idx = 0
#for i in range(len(boundary_neighbor_dir_lists)):
#    idx=i
#    gap_idc = np.nonzero(np.diff(boundary_neighbor_dir_lists[i])>1)[0]
#    if len(gap_idc) > 0:
#        assert(len(gap_idc==1))
#        gap_idx = gap_idc[0]
#        equivalent_dir = [boundary_neighbor_dir_lists[i][j]+8 if j<gap_idx+1 else boundary_neighbor_dir_lists[i][j] for j in range(len(boundary_neighbor_dir_lists[i]))]
#        mid = (min(equivalent_dir) + max(equivalent_dir))/2
#        mid = np.mod(mid,8)
#    else:
#        mid = (min(boundary_neighbor_dir_lists[i]) + max(boundary_neighbor_dir_lists[i]))/2
#    boundary_neighbor_mid_dir.append(mid)
#for i in range(len(boundary_neighbor_dir_lists)):
##    print(boundary_neighbor_lists[i], md_points[i]) 
#    assert(np.floor(boundary_neighbor_mid_dir[i]) in boundary_neighbor_dir_lists[i])
##    
#np.save('boundary_points',boundary_points)
#np.save('neighbor_points',neighbor_points)
#np.save('neighbor_dir_lists',neighbor_dir_lists)
#np.save('boundary_neighbor_dir_lists',boundary_neighbor_dir_lists)
#np.save('boundary_neighbor_mid_dir',boundary_neighbor_mid_dir)

# boundary_points = np.load('boundary_points.npy',allow_pickle=True)
# neighbor_points = np.load('neighbor_points.npy',allow_pickle=True)
# neighbor_dir_lists = np.load('neighbor_dir_lists.npy',allow_pickle=True)
# boundary_neighbor_dir_lists = np.load('boundary_neighbor_dir_lists.npy',allow_pickle=True)
# boundary_neighbor_mid_dir = np.load('boundary_neighbor_mid_dir.npy',allow_pickle=True)



                
#trajectory = []
#start_idx = np.random.randint(len(boundary_points))
#start =  boundary_points[start_idx]
#trajectory.append(start)
##available_dirs = np.nonzero(neighbor_points[start,:])[0]
##move_dir = np.random.choice(available_dirs)
#move_dir = int(np.floor(boundary_neighbor_mid_dir[start_idx]))
#current_point = start
#reach_boundary = False
#while True:
#    current_point = int(neighbor_points[current_point,move_dir]-1)
#    trajectory.append(current_point)
#    if current_point in boundary_points:
#        break
#    available_dirs = np.nonzero(neighbor_points[current_point,:])[0]
#    prob = np.zeros(8)
#    assert(move_dir in available_dirs)
#    assert(int(np.mod(move_dir-1,8)) in available_dirs)
#    assert(int(np.mod(move_dir+1,8)) in available_dirs)
#    prob[move_dir] =0.8
#    prob[int(np.mod(move_dir-1,8))] = 0.1
#    prob[int(np.mod(move_dir+1,8))] = 0.1
#    move_dir= np.random.choice(available_dirs,p=prob)    
#
#plt.scatter(all_loc[trajectory,0],all_loc[trajectory,1])
#print(len(trajectory))

#def gen_trajectory():
#    all_points = []
#    start = np.random.choice(boundary_points)
#    all_points.append(start)
#    available_dirs = np.nonzero(neighbor_points[start,:])
#    prev_dir = np.random.choice(np.nonzero(neighbor_points[start,:]))
#    turn_complete = True
#    end_of_trajectory = False
#    while not end_of_trajectory:
#        if turn_complete:

def gen_trajectory(neighbor_points,boundary_points,boundary_neighbor_mid_dir):
    trajectory = []
    start_idx = np.random.randint(len(boundary_points))
    start =  boundary_points[start_idx]
    trajectory.append(start)
    #available_dirs = np.nonzero(neighbor_points[start,:])[0]
    #move_dir = np.random.choice(available_dirs)
    move_dir = int(np.floor(boundary_neighbor_mid_dir[start_idx]))
    current_point = start
    while True:
        current_point = int(neighbor_points[current_point,move_dir]-1)
        trajectory.append(current_point)
        if current_point in boundary_points:
            break
        available_dirs = np.nonzero(neighbor_points[current_point,:])[0]
        prob = np.zeros(8)
        assert(move_dir in available_dirs)
        assert(int(np.mod(move_dir-1,8)) in available_dirs)
        assert(int(np.mod(move_dir+1,8)) in available_dirs)
        prob[move_dir] =0.95
        prob[int(np.mod(move_dir-1,8))] = 0.025
        prob[int(np.mod(move_dir+1,8))] = 0.025
        move_dir= np.random.choice(available_dirs,p=prob)    
    return trajectory 

#traj_len = [len(gen_trajectory()) for i in range(1000)]
#plt.hist(traj_len)
    
    
def DFT_codebook(oversampling_factor):
    n_antenna = 64
    nseg = int(n_antenna*oversampling_factor)
    bfdirections = np.arccos(np.linspace(np.cos(0),np.cos(np.pi-1e-6),nseg))
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    
    for i in range(nseg):
        phi = bfdirections[i]
        #array response vector original
        arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
        #array response vector for rotated ULA
        #arr_response_vec = [1j*np.pi*k*np.sin(phi+np.pi/2) for k in range(64)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all, bfdirections

class BeamManagementEnv(gym.Env):
    def __init__(self, oversampling_factor: int = 1,
             num_antennas : int = 64,
           num_beams_per_UE: int = 8,
           ue_speed = 5,
           enable_baseline = False,
           enable_genie = False,
           combine_state = False,
           full_observation = False,
           num_measurements = 1,
           min_traj_len = 5):
        self.enable_baseline = enable_baseline
        self.enable_genie = enable_genie
        self.combine_state = combine_state #flag for whether to include previous action in state representation: s(t)=[ob(t),a(t-1)]
        self.full_observation = full_observation # flag for whether to measure all beams in observation
        self.num_measurements = num_measurements #number of measurements per timestep: if 1, 1 measurements collected at the end of the period. otherwise sample in equal partitions
        self.n_antenna = num_antennas
        self.oversampling_factor = oversampling_factor
        self.codebook_size = int(self.n_antenna*self.oversampling_factor)
        self.num_beams_per_UE = num_beams_per_UE
        self.action_space = spaces.MultiBinary(self.codebook_size)
        self.min_traj_len = min_traj_len
        if self.combine_state:
            self.observation_space = spaces.Box(low = np.full(int(2*self.codebook_size),-np.inf), high = np.full(int(2*self.codebook_size),np.inf), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low = np.full(self.codebook_size,-np.inf), high = np.full(self.codebook_size,np.inf), dtype=np.float32)
#        self.observation_space = spaces.MultiDiscrete(np.inf*np.ones(self.codebook_size))
        self.true_state = np.zeros((self.codebook_size))
        
        # self.h = np.load(h_real_fname) + 1j*np.load(h_imag_fname)
        self._load_channels()
        self.ue_loc = np.load(ue_loc_fname)[:,0:2]
        self.boundary_points = np.load('boundary_points.npy',allow_pickle=True)
        self.neighbor_points = np.load('neighbor_points.npy',allow_pickle=True)
        self.neighbor_dir_lists = np.load('neighbor_dir_lists.npy',allow_pickle=True)
        self.boundary_neighbor_dir_lists = np.load('boundary_neighbor_dir_lists.npy',allow_pickle=True)
        self.boundary_neighbor_mid_dir = np.load('boundary_neighbor_mid_dir.npy',allow_pickle=True)
        #        self.unique_x = np.unique(self.ue_loc[:,0])
        #        self.unique_y = np.unique(self.ue_loc[:,1])
        # self.codebook_all, self.bf_directions = DFT_codebook(self.oversampling_factor)
        self.codebook_all, self.bf_directions = self._generate_DFT_codebook()
               
        self.t = 0
#        self.n_current_UEs = 0
        self.traj = []
        self.current_UE_pos = 0
        self.assigned_beams_per_UE = []
        # self.current_state_single_frame = []
        self.ue_speed = ue_speed
        self.prev_info = {}
        self.reward_log = {}
        self.seed()
        self.use_saved_trajectory = False
        self.num_saved_traj = None
        
    def set_data_mode(self, use_saved_trajectory=False, num_saved_traj=None):
        self.use_saved_trajectory = use_saved_trajectory
        if use_saved_trajectory:
            assert(not num_saved_traj is None)
        self.num_saved_traj = num_saved_traj
        if self.use_saved_trajectory:
            self.saved_traj= np.load('saved_traj.npy',allow_pickle=True)[0:self.num_saved_traj]
            self.saved_traj_pos = np.load('saved_traj_pos.npy',allow_pickle=True)[0:self.num_saved_traj]
            self.saved_traj_edge_lengths = np.load('saved_traj_edge_lengths.npy',allow_pickle=True)[0:self.num_saved_traj]
            self.saved_traj_point_distances = np.load('saved_traj_point_distances.npy',allow_pickle=True)[0:self.num_saved_traj]
            self.saved_traj_total_len = np.load('saved_traj_total_len.npy',allow_pickle=True)[0:self.num_saved_traj]
            self.get_saved_traj_idx = 0
        else:
            self.saved_traj= None
            self.saved_traj_pos = None
            self.saved_traj_edge_lengths = None
            self.saved_traj_point_distances = None
            self.saved_traj_total_len = None
            self.get_saved_traj_idx = None   

    def get_trajectory(self):
        if self.use_saved_trajectory:
            traj = self.saved_traj[self.get_saved_traj_idx]
            traj_pos = self.saved_traj_pos[self.get_saved_traj_idx]
            traj_edge_lengths = self.saved_traj_edge_lengths[self.get_saved_traj_idx]
            traj_total_len = self.saved_traj_total_len[self.get_saved_traj_idx]
            traj_point_distances = self.saved_traj_point_distances[self.get_saved_traj_idx]
            self.get_saved_traj_idx = (self.get_saved_traj_idx + 1) % self.num_saved_traj
        else:
            while True:
                traj = gen_trajectory(self.neighbor_points, self.boundary_points,self.boundary_neighbor_mid_dir)
                traj = np.array(traj)
    #            if len(traj)>100*self.ue_speed/0.35:
                traj_pos = self.ue_loc[traj]
                traj_edge_lengths = np.linalg.norm(np.diff(traj_pos, axis=0),axis=1)
                traj_total_len = sum(traj_edge_lengths)
                traj_point_distances = np.insert(np.cumsum(traj_edge_lengths) ,0,0)
                # if traj_total_len > (self.window_length+1)*self.ue_speed:
                if traj_total_len > self.ue_speed*self.min_traj_len:
                    break
        return traj, traj_pos, traj_edge_lengths, traj_total_len, traj_point_distances
            
    # def get_trajectory(self):
    #     while True:
    #         traj = gen_trajectory(self.neighbor_points, self.boundary_points,self.boundary_neighbor_mid_dir)
    #         traj = np.array(traj)
    #         traj_pos = self.ue_loc[traj]
    #         traj_edge_lengths = np.linalg.norm(np.diff(traj_pos, axis=0),axis=1)
    #         traj_total_len = sum(traj_edge_lengths)
    #         traj_point_distances = np.insert(np.cumsum(traj_edge_lengths) ,0,0)
    #         if traj_total_len > self.ue_speed*5:
    #             return traj, traj_pos, traj_edge_lengths, traj_total_len, traj_point_distances
            
#     def get_trajectory(self):
#         while True:
#             traj = gen_trajectory(self.neighbor_points, self.boundary_points,self.boundary_neighbor_mid_dir)
# #            if len(traj)>100*self.ue_speed/0.35:
#             if len(traj)>100:
#                 return np.array(traj)
    
    def measure_beams_single_UE(self,ue_pos, beam_idc):
        #return the SNR (dB) of selected beams in beam_idc (array) using the h matrix at ue_pos (global index for h)
        result = np.absolute(np.matmul(self.h[ue_pos,:], np.transpose(np.conj(self.codebook_all[beam_idc]))))**2
        return 30 + 10*np.log10(result)-(-94)
                
    def step(self, action): #action is codebook_size x 1 mulit bineary vector indicting which beams are activated
        assert(len(np.nonzero(action)[0])==self.num_beams_per_UE)
        self.assigned_beams_per_UE = np.nonzero(action)[0]
        # self.assigned_beams_per_UE = np.argsort(action)[-self.num_beams_per_UE:]
        info = {}
        #get reward
        ue_traveled_dist_next = (self.t+1)*self.ue_speed
        episode_end = False
        if ue_traveled_dist_next >= max(self.traj_point_distances):
            ue_traveled_dist_next = max(self.traj_point_distances)
            episode_end = True
        total_segment_length = ue_traveled_dist_next - self.ue_traveled_distance
        idc_in_traj_covered = np.nonzero(np.logical_and(self.traj_point_distances > self.ue_traveled_distance, self.traj_point_distances < ue_traveled_dist_next))[0] #idc of points in traj in the segment
#        print(np.nonzero(self.traj_point_distances > self.ue_traveled_distance)[0])
        # if len(np.nonzero(self.traj_point_distances > self.ue_traveled_distance)[0]) == 0:
        #     print('ohh')
        prev_h_idc_in_traj = min(np.nonzero(self.traj_point_distances > self.ue_traveled_distance)[0])-1
        h_idc_in_traj_covered = np.insert(idc_in_traj_covered,0,prev_h_idc_in_traj) #idc of h (local idc) in traj, including starting h (last point in prev segment)
        h_idc_covered = self.traj[h_idc_in_traj_covered] #global idc
        if len(h_idc_covered) == 1:
            # H doesnt change within the segment
            segment_bf_gains = self.measure_beams_single_UE(h_idc_covered[0], self.assigned_beams_per_UE)
            reward = np.log2(1+10**(max(segment_bf_gains)/10))
        else:
            reward = 0
            for i in range(len(h_idc_covered)):
                segment_bf_gains = self.measure_beams_single_UE(h_idc_covered[i], self.assigned_beams_per_UE)
                segment_achievable_rate = np.log2(1+10**(max(segment_bf_gains)/10))
                if i == 0:
                    segment_end = self.traj_point_distances[h_idc_in_traj_covered[i+1]]
                    segment_start = self.ue_traveled_distance
                if i == len(h_idc_covered)-1:
                    segment_end = ue_traveled_dist_next
                    segment_start = self.traj_point_distances[h_idc_in_traj_covered[i]]
                if i > 0 and i < len(h_idc_covered)-1:
                    segment_start = self.traj_point_distances[h_idc_in_traj_covered[i]]
                    segment_end = self.traj_point_distances[h_idc_in_traj_covered[i+1]]
                reward += segment_achievable_rate*(segment_end - segment_start)/total_segment_length  
        self.prev_info['agent_reward'] = reward
        self.reward_log['agent'].append(reward)

        if self.enable_baseline:
            if len(h_idc_covered) == 1:
                # H doesnt change within the segment
                baseline_segment_bf_gains = self.measure_beams_single_UE(h_idc_covered[0], self.baseline_beams)
                baseline_reward = np.log2(1+10**(max(baseline_segment_bf_gains)/10))
            else:
                baseline_reward = 0
                for i in range(len(h_idc_covered)):
                    baseline_segment_bf_gains = self.measure_beams_single_UE(h_idc_covered[i], self.baseline_beams)
                    baseline_segment_achievable_rate = np.log2(1+10**(max(baseline_segment_bf_gains)/10))
                    if i == 0:
                        segment_end = self.traj_point_distances[h_idc_in_traj_covered[i+1]]
                        segment_start = self.ue_traveled_distance
                    if i == len(h_idc_covered)-1:
                        segment_end = ue_traveled_dist_next
                        segment_start = self.traj_point_distances[h_idc_in_traj_covered[i]]
                    if i > 0 and i < len(h_idc_covered)-1:
                        segment_start = self.traj_point_distances[h_idc_in_traj_covered[i]]
                        segment_end = self.traj_point_distances[h_idc_in_traj_covered[i+1]]
                    baseline_reward += baseline_segment_achievable_rate*(segment_end - segment_start)/total_segment_length      
            info['baseline_reward'] = baseline_reward
            self.prev_info['baseline_reward'] = baseline_reward
            self.reward_log['baseline'].append(baseline_reward)
            info['baseline_beams'] = self.baseline_beams
        
        if self.enable_genie:
            info['genie_beams'] = []
            if len(h_idc_covered) == 1:
                # H doesnt change within the segment
                genie_segment_bf_gains = self.measure_beams_single_UE(h_idc_covered[0], np.arange(self.codebook_size))
                genie_reward = np.log2(1+10**(max(genie_segment_bf_gains)/10))
                genie_beam = np.argmax(genie_segment_bf_gains)
                info['genie_beams'].append(genie_beam)
            else:
                genie_reward = 0
                for i in range(len(h_idc_covered)):
                    genie_segment_bf_gains = self.measure_beams_single_UE(h_idc_covered[i], np.arange(self.codebook_size))
                    genie_segment_achievable_rate = np.log2(1+10**(max(genie_segment_bf_gains)/10))
                    info['genie_beams'].append(np.argmax(genie_segment_bf_gains))
                    if i == 0:
                        segment_end = self.traj_point_distances[h_idc_in_traj_covered[i+1]]
                        segment_start = self.ue_traveled_distance
                    if i == len(h_idc_covered)-1:
                        segment_end = ue_traveled_dist_next
                        segment_start = self.traj_point_distances[h_idc_in_traj_covered[i]]
                    if i > 0 and i < len(h_idc_covered)-1:
                        segment_start = self.traj_point_distances[h_idc_in_traj_covered[i]]
                        segment_end = self.traj_point_distances[h_idc_in_traj_covered[i+1]]
                    genie_reward += genie_segment_achievable_rate*(segment_end - segment_start)/total_segment_length      
            info['genie_reward'] = genie_reward
            info['genie_beams'] = np.unique(info['genie_beams'])
            self.prev_info['genie_reward'] = genie_reward
            self.reward_log['genie'].append(genie_reward)
            
                     
        self._time_increment()
        # assigned_bf_gains = self.measure_beams_single_UE(self.current_h_idc, self.assigned_beams_per_UE)
        # beam_report = np.zeros((self.codebook_size))
        # beam_report[self.assigned_beams_per_UE] = assigned_bf_gains
        
        # update baseline beams (used in next time step) that's centered around the best beam in the last h in this segment
        if self.enable_baseline:
            baseline_max_beam = self.baseline_beams[np.argmax(self.measure_beams_single_UE(self.current_h_idc,self.baseline_beams))]
            self.baseline_beams = self.calc_baseline_beams(baseline_max_beam)   
        
        # if self.combine_state:
        #     observation = np.concatenate((beam_report,action),axis=0)
        # else:
        #     observation = beam_report
        if self.full_observation:
            observation = self._get_full_observation()
        else:
            observation = self._get_observation()
        
        return observation, reward, episode_end, info
    
    def render(self):
        print(self.prev_info)
        
    def set_ue_speed(self,speed):
        self.ue_speed = speed

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def get_initial_beam_assignment(self):
        bf_gains = self.measure_beams_single_UE(self.current_h_idc,np.arange(self.codebook_size))
        best_beams = np.argsort(bf_gains)[-self.num_beams_per_UE:]
        best_bf_gains = bf_gains[best_beams]
        return best_beams, best_bf_gains
    
    def calc_baseline_beams(self, baseline_max_beam):
        #calculate the moving spatial window beamset
        if baseline_max_beam < self.num_beams_per_UE/2:
            baseline_beamset_start = 0
            baseline_beamset_end = baseline_beamset_start+self.num_beams_per_UE
        elif baseline_max_beam > self.codebook_size-self.num_beams_per_UE/2:
            baseline_beamset_end = self.codebook_size
            baseline_beamset_start = baseline_beamset_end-self.num_beams_per_UE
        else:
            baseline_beamset_start = baseline_max_beam-self.num_beams_per_UE/2
            baseline_beamset_end = baseline_max_beam+self.num_beams_per_UE/2
        baseline_beamset = np.ceil(np.arange(baseline_beamset_start,baseline_beamset_end)).astype(int)
        assert(len(baseline_beamset) == self.num_beams_per_UE)    
        return baseline_beamset
    
    # def calc_baseline_beams(self, baseline_max_beam):
    #     #calculate the moving spatial window beamset
    #     if baseline_max_beam < 3:
    #         baseline_beamset_start = 0
    #         baseline_beamset_end = baseline_beamset_start+self.num_beams_per_UE
    #     elif baseline_max_beam > self.codebook_size-5:
    #         baseline_beamset_end = self.codebook_size
    #         baseline_beamset_start = baseline_beamset_end-self.num_beams_per_UE
    #     else:
    #         baseline_beamset_start = baseline_max_beam-3
    #         baseline_beamset_end = baseline_max_beam+5
    #     baseline_beamset = np.arange(baseline_beamset_start,baseline_beamset_end)
    #     assert(len(baseline_beamset) == self.num_beams_per_UE)    
    #     return baseline_beamset
    
    def reset(self):
        # self.ue_speed = 5 
#        self.n_current_UEs = 0
        self.reward_log = {}
        self.reward_log['agent'] = []
        if self.enable_baseline:
            self.reward_log['baseline'] = []
        if self.enable_genie:
            self.reward_log['genie'] = []
        
        self.traj, self.traj_pos, self.traj_edge_lengths, self.traj_total_len, self.traj_point_distances = self.get_trajectory()
        # self.traj = self.get_trajectory()
        # self.traj_pos = self.ue_loc[self.traj]
        # self.traj_edge_lengths = np.linalg.norm(np.diff(self.traj_pos, axis=0),axis=1)
        # self.traj_point_distances = np.insert(np.cumsum(self.traj_edge_lengths) ,0,0)
        self.t = 0
        self.ue_traveled_distance = self.t*self.ue_speed
        self.current_idc_in_traj = max(np.nonzero(self.traj_point_distances <= self.ue_traveled_distance)[0])
        self.current_h_idc = self.traj[self.current_idc_in_traj]
        self.assigned_beams_per_UE, assigned_bf_gains = self.get_initial_beam_assignment() #assign initial beams with genie action
                
        # initial_beams = np.zeros((self.codebook_size))
        # initial_beams[self.assigned_beams_per_UE] = 1
        if self.enable_baseline:
            baseline_max_beam = np.argmax(self.measure_beams_single_UE(self.current_h_idc,np.arange(self.codebook_size)))
            baseline_beamset = self.calc_baseline_beams(baseline_max_beam)
            self.baseline_beams = baseline_beamset
        
        # """
        # # if num_measuremets > 1, measure beams in between:
        # #     initial beamset is an action appplied at t=0, collect measurements from t=0 to 1
        # # otherwise, measure beams at the last traj point before t=1
        # """
        # if self.num_measurements > 1:
        #     beam_report = np.zeros((self.num_measurements, self.codebook_size))
        #     for temp_t_idx, temp_t in np.enumerate(np.arange(self.t, self.t+1, 1/self.num_measurements)):
        #         temp_traveld_distance = temp_t * self.ue_speed
        #         temp_idc_in_traj = max(np.nonzero(self.traj_point_distances <= temp_traveld_distance)[0])
        #         temp_current_h_idc = self.traj[temp_idc_in_traj]
        #         temp_beam_measurements = self.measure_beams_single_UE(temp_current_h_idc,self.assigned_beams_per_UE)
        #         beam_report[self.assigned_beams_per_UE] = temp_beam_measurements
        #     if self.combine_state:
        #         action_matrix = np.zeros((self.num_measurements,self.codebook_size))
        #         action_matrix[:,self.assigned_beams_per_UE] = 1
        #         observation = np.concatenate((beam_report,action_matrix),axis=1)
        #     else:
        #         observation = beam_report               
        # else:
        #     beam_report = np.zeros((self.codebook_size))
        #     next_ue_traveled_distance = (self.t+1)*self.ue_speed
        #     last_idc_in_traj_in_segment = max(np.nonzero(self.traj_point_distances < next_ue_traveled_distance)[0])
        #     last_h_idc_global_in_segment = self.traj[last_idc_in_traj_in_segment]
        #     beam_report[self.assigned_beams_per_UE] = measure_beams_single_UE(last_h_idc_global_in_segment, self.assigned_beams_per_UE):
        #     # self.current_state_single_frame = beam_report
        #     if self.combine_state:
        #         observation = np.concatenate((beam_report,initial_beams),axis=0)
        #     else:
        #         observation = beam_report
        
        if self.full_observation:
            observation = self._get_full_observation()
        else:
            observation = self._get_observation()

        self._time_increment()
        # prev_h_idc_in_traj = min(np.nonzero(self.traj_point_distances > self.ue_traveled_distance)[0])-1
        # prev_h_idc_global = self.traj[prev_h_idc_in_traj]
        # assigned_bf_gains = self.measure_beams_single_UE(self.current_h_idc, self.assigned_beams_per_UE)
        # beam_report = np.zeros((self.codebook_size))
        # beam_report[self.assigned_beams_per_UE] = assigned_bf_gains
        if self.enable_baseline:
            # baseline_max_beam = self.baseline_beams[np.argmax(self.measure_beams_single_UE(self.current_h_idc,self.baseline_beams))]
            baseline_max_beam = np.argmax(self.measure_beams_single_UE(self.current_h_idc,self.assigned_beams_per_UE))
            self.baseline_beams = self.calc_baseline_beams(baseline_max_beam)   
            
        return observation
    
    def _time_increment(self):
        self.t += 1
        self.ue_traveled_distance = self.t*self.ue_speed
        self.current_idc_in_traj = max(np.nonzero(self.traj_point_distances < self.ue_traveled_distance)[0])
        self.current_h_idc = self.traj[self.current_idc_in_traj]
        
    def _get_observation(self):
        """
        # if num_measuremets > 1, measure beams in between:
        #     initial beamset is an action appplied at t=0, collect measurements from t=0 to 1
        # otherwise, measure beams at the last traj point before t=1
        """
        if self.num_measurements > 1:
            beam_report = np.zeros((self.num_measurements, self.codebook_size))
            for temp_t_idx, temp_t in enumerate(np.arange(self.t, self.t+1, 1/self.num_measurements)):
                temp_traveld_distance = temp_t * self.ue_speed
                temp_idc_in_traj = max(np.nonzero(self.traj_point_distances <= temp_traveld_distance)[0])
                temp_current_h_idc = self.traj[temp_idc_in_traj]
                temp_beam_measurements = self.measure_beams_single_UE(temp_current_h_idc,self.assigned_beams_per_UE)
                beam_report[temp_t_idx, self.assigned_beams_per_UE] = temp_beam_measurements
            if self.combine_state:
                action_matrix = np.zeros((self.num_measurements,self.codebook_size))
                action_matrix[:,self.assigned_beams_per_UE] = 1
                observation = np.concatenate((beam_report,action_matrix),axis=1)
            else:
                observation = beam_report               
        else:
            beam_report = np.zeros((self.codebook_size))
            next_ue_traveled_distance = (self.t+1)*self.ue_speed
            last_idc_in_traj_in_segment = max(np.nonzero(self.traj_point_distances < next_ue_traveled_distance)[0])
            last_h_idc_global_in_segment = self.traj[last_idc_in_traj_in_segment]
            beam_report[self.assigned_beams_per_UE] = self.measure_beams_single_UE(last_h_idc_global_in_segment, self.assigned_beams_per_UE)
            # self.current_state_single_frame = beam_report
            if self.combine_state:
                binary_action_vector = np.zeros((self.codebook_size))
                binary_action_vector[self.assigned_beams_per_UE] = 1
                observation = np.concatenate((beam_report,binary_action_vector),axis=0)
            else:
                observation = beam_report      
        return observation

    def _get_full_observation(self):
        """
        # if num_measuremets > 1, measure beams in between:
        #     initial beamset is an action appplied at t=0, collect measurements from t=0 to 1
        # otherwise, measure beams at the last traj point before t=1
        """
        if self.num_measurements > 1:
            beam_report = np.zeros((self.num_measurements, self.codebook_size))
            for temp_t_idx, temp_t in enumerate(np.arange(self.t, self.t+1, 1/self.num_measurements)):
                temp_traveld_distance = temp_t * self.ue_speed
                temp_idc_in_traj = max(np.nonzero(self.traj_point_distances <= temp_traveld_distance)[0])
                temp_current_h_idc = self.traj[temp_idc_in_traj]
                temp_beam_measurements = self.measure_beams_single_UE(temp_current_h_idc,np.arange(self.codebook_size))
                beam_report[temp_t_idx, :] = temp_beam_measurements
            if self.combine_state:
                action_matrix = np.zeros((self.num_measurements,self.codebook_size))
                action_matrix[:,self.assigned_beams_per_UE] = 1
                observation = np.concatenate((beam_report,action_matrix),axis=1)
            else:
                observation = beam_report               
        else:
            beam_report = np.zeros((self.codebook_size))
            next_ue_traveled_distance = (self.t+1)*self.ue_speed
            last_idc_in_traj_in_segment = max(np.nonzero(self.traj_point_distances < next_ue_traveled_distance)[0])
            last_h_idc_global_in_segment = self.traj[last_idc_in_traj_in_segment]
            beam_report[self.assigned_beams_per_UE] = self.measure_beams_single_UE(last_h_idc_global_in_segment, np.arange(self.codebook_size))
            # self.current_state_single_frame = beam_report
            if self.combine_state:
                binary_action_vector = np.zeros((self.codebook_size))
                binary_action_vector[self.assigned_beams_per_UE] = 1
                observation = np.concatenate((beam_report,binary_action_vector),axis=0)
            else:
                observation = beam_report      
        return observation
    
    def _load_channels(self):
        all_h = np.load(h_real_fname) + 1j*np.load(h_imag_fname)
        self.num_ant_total = all_h.shape[1]
        assert(np.mod(self.num_ant_total, self.n_antenna)==0)
        antenna_step = int(self.num_ant_total/self.n_antenna)
        subsampled_antennas = np.arange(0,self.num_ant_total,antenna_step)
        assert(len(subsampled_antennas)==self.n_antenna)
        self.h = all_h[:,subsampled_antennas]
            
    def _generate_DFT_codebook(self):
        nseg = int(self.n_antenna*self.oversampling_factor)
        bfdirections = np.arccos(np.linspace(np.cos(0),np.cos(np.pi-1e-6),nseg))
        codebook_all = np.zeros((nseg,self.n_antenna),dtype=np.complex_)
        default_spacing = 0.5
        spacing = default_spacing * self.num_ant_total / self.n_antenna
        for i in range(nseg):
            phi = bfdirections[i]
            #array response vector original
            arr_response_vec = [-1j*2*np.pi*spacing*k*np.cos(phi) for k in range(self.n_antenna)]
            # arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
            #array response vector for rotated ULA
            #arr_response_vec = [1j*np.pi*k*np.sin(phi+np.pi/2) for k in range(64)]
            codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(self.n_antenna)
        return codebook_all, bfdirections

        
 
# if __name__ == "__main__":
#     env = BeamManagementEnv(num_antennas=32)    

# if __name__ == "__main__":
#     env = BeamManagementEnv()
#     env.set_data_mode(use_saved_trajectory=True,num_saved_traj=50)
#     env.reset()
#     while True:
#         action = np.random.choice(np.arange(env.codebook_size),env.num_beams_per_UE,replace=False)
#         binary_action = np.zeros((env.codebook_size))
#         binary_action[action] = 1
#         s,r,done,info = env.step(binary_action)
#         if done:
#             print(env.get_saved_traj_idx)
#             env.reset()
#             if env.get_saved_traj_idx == 0:
#                 env.set_data_mode(use_saved_trajectory=False)
#                 print(env.get_saved_traj_idx)
#                 env.reset()
#                 while True:
#                     action = np.random.choice(np.arange(env.codebook_size),env.num_beams_per_UE,replace=False)
#                     binary_action = np.zeros((env.codebook_size))
#                     binary_action[action] = 1
#                     st,rt,donet,infot = env.step(binary_action)
#                     if donet:
#                         print('finished random episode')
#                         break
#                 break
            

# from tqdm import tqdm

# if __name__ == "__main__":
#     num_trajs = int(1e6)
#     env = BeamManagementEnv(ue_speed=15)
#     data = {'traj':[],'traj_pos':[],'traj_edge_lengths':[],'traj_total_len':[],'traj_point_distances':[]}
#     for i in tqdm(range(num_trajs)):
#         traj, traj_pos, traj_edge_lengths, traj_total_len, traj_point_distances = env.get_trajectory()
#         data['traj'].append(traj)
#         data['traj_pos'].append(traj_pos)
#         data['traj_edge_lengths'].append(traj_edge_lengths)
#         data['traj_total_len'].append(traj_total_len)
#         data['traj_point_distances'].append(traj_point_distances)
#     np.save('saved_traj',data['traj'])
#     np.save('saved_traj_pos',data['traj_pos'])
#     np.save('saved_traj_edge_lengths',data['traj_edge_lengths'])
#     np.save('saved_traj_total_len',data['traj_total_len'])
#     np.save('saved_traj_point_distances',data['traj_point_distances'])
    
#from tqdm import tqdm     
# import matplotlib.pyplot as plt
# if __name__ == "__main__":
#     env = BeamManagementEnv(enable_baseline=False, enable_genie=True, ue_speed=15, num_measurements = 5, combine_state = False, num_beams_per_UE = 64)
#     num_beams = env.codebook_size
#     s = env.reset()
#     prev_s=s
#     plt.figure()
#     plt.plot(env.ue_loc[env.traj][:,0],env.ue_loc[env.traj][:,1])
#     s_history = []
#     s_history.append(s)
#     done = False
#     while not done:
#         # beams = np.random.choice(np.arange(num_beams),8,replace=False)
#         # beam_config = np.zeros((num_beams)).astype(int)
#         # beam_config[beams] = 1
#         beam_config = np.ones((num_beams))
#         s_t, r_t, done, info = env.step(beam_config)
#         s_history.append(s_t)
#         if not done:
#             prev_s = s_t
# #        print(r_t)
#         # print(info['genie_reward']-info['baseline_reward'])
# #        print(info['baseline_beams'])
#     for i in range(len(s_history)):
#         if i == 0:
#             s_arr = s_history[i]
#         else:
#             s_arr = np.concatenate((s_arr,s_history[i]),axis=0)

# import gym

# from keras.models import Sequential, Model
# from keras.layers import Dense, Activation, Flatten, Input, Concatenate
# from keras.optimizers import Adam
# from keras.layers import Lambda
# import tensorflow as tf
# #from rl.agents import DDPGAgent
# from modified_DDPG import modified_DDPGAgent
# from rl.memory import SequentialMemory
# from rl.random import OrnsteinUhlenbeckProcess

# def top_k(input, k):
#     return tf.nn.top_k(input, k=k, sorted=True).indices

# if __name__ == "__main__":
#     # Get the environment and extract the number of actions.
#     env = BeamManagementEnv(oversampling_factor = 2)
#     np.random.seed(123)
#     env.seed(123)
#     assert len(env.action_space.shape) == 1
#     nb_actions = env.action_space.shape[0]
    
#     # Next, we build a very simple model.
#     actor = Sequential()
#     actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
#     actor.add(Dense(64))
#     actor.add(Activation('relu'))
#     actor.add(Dense(32))
#     actor.add(Activation('relu'))
#     actor.add(Dense(32))
#     actor.add(Activation('relu'))
#     actor.add(Dense(nb_actions))
#     actor.add(Activation('linear'))
# #    actor.add(Lambda(top_k, arguments={'k': 8}))
#     print(actor.summary())
    
#     action_input = Input(shape=(nb_actions,), name='action_input')
#     observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
#     flattened_observation = Flatten()(observation_input)
#     x = Concatenate()([action_input, flattened_observation])
#     x = Dense(32)(x)
#     x = Activation('relu')(x)
#     x = Dense(32)(x)
#     x = Activation('relu')(x)
#     x = Dense(32)(x)
#     x = Activation('relu')(x)
#     x = Dense(1)(x)
#     x = Activation('linear')(x)
#     critic = Model(inputs=[action_input, observation_input], outputs=x)
#     print(critic.summary())
    
#     # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
#     # even the metrics!
#     memory = SequentialMemory(limit=100000, window_length=1)
#     random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
#     agent = modified_DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
#                       memory=memory, nb_steps_warmup_critic=10, nb_steps_warmup_actor=10,
#                       random_process=random_process, gamma=.99, target_model_update=1e-3)
#     agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    
#     # Okay, now it's time to learn something! We visualize the training here for show, but this
#     # slows down training quite a lot. You can always safely abort the training prematurely using
#     # Ctrl + C.
#     history = agent.fit(env, nb_steps=50000, visualize=False, verbose=1, nb_max_episode_steps=200)
    
#     # After training is done, we save the final weights.
#     agent.save_weights('ddpg_{}_weights.h5f'.format('BeamManagementEnv'), overwrite=True)
    
#     # Finally, evaluate our algorithm for 5 episodes.
#     agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)
    
        
    
