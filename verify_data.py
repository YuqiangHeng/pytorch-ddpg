# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:18:20 2019

@author: (Ethan) Yuqiang Heng
"""
import numpy as np
import matplotlib.pyplot as plt

def cart2sph(xyz,center):
    x = np.subtract(xyz[:,0],center[0])
    y = np.subtract(xyz[:,1],center[1])
    z = np.subtract(xyz[:,2],center[2])
    rtp = np.zeros(xyz.shape)
    r = np.sqrt(np.power(x,2)+np.power(y,2)+np.power(z,2))
    theta = np.arccos(np.divide(z,r))
    #phi = np.arctan(np.divide(y,x))
    phi = np.arctan2(y,x)
    rtp[:,0] = r
    rtp[:,1] = theta
    rtp[:,2] = phi
    return rtp

h_imag_fname = "H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_imag.npy"
h_real_fname = "H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_real.npy"
ue_loc_fname = "H_Matrices FineGrid/MISO_Static_FineGrid_UE_location.npy"
bs_loc = [641,435,10]
n_antenna =8
oversampling_factor = 1

all_h = np.load(h_imag_fname)*1j+np.load(h_real_fname)
all_loc = np.load(ue_loc_fname)
all_sph = cart2sph(all_loc, bs_loc)

num_ant_total = all_h.shape[1]
assert(np.mod(num_ant_total, n_antenna)==0)
antenna_step = int(num_ant_total/n_antenna)
subsampled_antennas = np.arange(0,num_ant_total,antenna_step)
assert(len(subsampled_antennas)==n_antenna)
h = all_h[:,subsampled_antennas]
        

nseg = int(n_antenna*oversampling_factor)
bfdirections = np.arccos(np.linspace(np.cos(0),np.cos(np.pi-1e-6),nseg))
codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
default_spacing = 0.5
spacing = default_spacing * num_ant_total / n_antenna
for i in range(nseg):
    phi = bfdirections[i]
    #array response vector original
    arr_response_vec = [-1j*2*np.pi*spacing*k*np.cos(phi) for k in range(n_antenna)]
    # arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
    #array response vector for rotated ULA
    #arr_response_vec = [1j*np.pi*k*np.sin(phi+np.pi/2) for k in range(64)]
    codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)

    




bf_gains = np.absolute(np.matmul(h, np.transpose(np.conj(codebook_all))))**2 #shape n_ue x codebook_size
all_snr = 30+10*np.log10(bf_gains)-(-94)

max_beams = np.argmax(all_snr, axis = 1)
max_beam_dir = bfdirections[max_beams]
plt.figure()
plt.scatter(all_sph[:,2],max_beam_dir)

for plt_idx in np.random.choice(all_h.shape[0],10).astype(int):
    plt.figure()
    plt.plot(bfdirections,all_snr[plt_idx,:])
    plt.axvline(x = max_beam_dir[plt_idx])
