'''
This file iterates over each annotated recordings and computes the metrics. 

These metrics are stored in a dictionnary with the state as key and a list 
of the metrics values for this given state
'''

import numpy as np
import scipy as sc
import os
from state_annotation.compute import Compute

# function to smooth tthe metrics based on history of 3
def smooth_last3(arr):
    """Smooth a 1D array using the average of the last 3 values (including current).
    Handles edge cases at the start."""
    arr = np.asarray(arr)
    smoothed = np.zeros_like(arr, dtype=float)
    
    for i in range(len(arr)):
        if i == 0:
            smoothed[i] = arr[i]
        elif i == 1:
            smoothed[i] = np.mean(arr[i-1:i+1])
        else:
            smoothed[i] = np.mean(arr[i-2:i+1])
    
    return smoothed

#--- initializes dictionnaries to store data
D_prop_delta = {i: [] for i in range(22)}
D_prop_alpha = {i: [] for i in range(22)}
D_prop_beta = {i: [] for i in range(22)}
D_prop_gamma = {i: [] for i in range(22)}
D_alpha_delta = {i: [] for i in range(22)}
D_beta_delta = {i: [] for i in range(22)}
D_gamma_delta = {i: [] for i in range(22)}
D_beta_alpha = {i: [] for i in range(22)}
D_gamma_alpha = {i: [] for i in range(22)}
D_gamma_beta = {i: [] for i in range(22)}
D_hf_lf = {i: [] for i in range(22)}
D_50_q = {i: [] for i in range(22)}
D_75_q = {i: [] for i in range(22)}
D_85_q = {i: [] for i in range(22)}
D_95_q = {i: [] for i in range(22)}
D_supp = {i: [] for i in range(22)}
D_line_length = {i: [] for i in range(22)}
D_entropy = {i: [] for i in range(22)}
D_be = {i: [] for i in range(22)}
D_f_central = {i: [] for i in range(22)}


#--- iterate over each recordings
folder_path = 'data_state_annotation'  
# List all elements (files and folders)
elements = os.listdir(folder_path)

for elem in elements:
    try:
        name = elem[2:]
        print(name)

        #--- load data
        data_states = np.load('data_state_annotation/D_' + name, allow_pickle=True).item()

        y = np.load('recordings_npy/' + name)
        N = len(y)
        fs = 128
        t = np.linspace(0, N/fs, N)

        #--- get variables
        C = Compute()
        C.get_data(t, y, fs, Ws = 30 *fs, step = 10 * fs, Ws_line_length = 30 * fs, step_line_length = 10 * fs)
        C.run()

        #--- smooth data
        # C.prop_P_signals[0,:] = smooth_last3(C.prop_P_signals[0,:])
        # C.prop_P_signals[1,:] = smooth_last3(C.prop_P_signals[1,:])
        # C.prop_P_signals[2,:] = smooth_last3(C.prop_P_signals[2,:])
        # C.prop_P_signals[1,:] = smooth_last3(C.prop_P_signals[-1,:])
        # C.freqs_quantiles[0,:] = smooth_last3(C.freqs_quantiles[0,:])
        # C.freqs_quantiles[1,:] = smooth_last3(C.freqs_quantiles[1,:])
        # C.freqs_quantiles[2,:] = smooth_last3(C.freqs_quantiles[2,:])
        # C.freqs_quantiles[-1,:] = smooth_last3(C.freqs_quantiles[-1,:])
        # C.supp = smooth_last3(C.supp)
        # C.line_length = smooth_last3(C.line_length)
        # C.entropy = smooth_last3(C.entropy)
        # C.be = smooth_last3(C.be)
        # C.f_central = smooth_last3(C.f_central)

        #--- ratio
        alpha_delta = C.prop_P_signals[1,:] / C.prop_P_signals[0,:]
        beta_delta = C.prop_P_signals[2,:] / C.prop_P_signals[0,:]
        gamma_delta = C.prop_P_signals[-1,:] / C.prop_P_signals[0,:]
        beta_alpha = C.prop_P_signals[2,:] / C.prop_P_signals[1,:]
        gamma_alpha = C.prop_P_signals[-1,:] / C.prop_P_signals[1,:]
        gamma_beta = C.prop_P_signals[-1,:] / C.prop_P_signals[2,:]
        hf_lf = (C.prop_P_signals[-1,:] + C.prop_P_signals[2,:]) / (C.prop_P_signals[1,:] + C.prop_P_signals[0,:])

        for i in range(22):
            mask = data_states['state_updated'] == i
            D_prop_delta[i].extend(C.prop_P_signals[0,:][mask])
            D_prop_alpha[i].extend(C.prop_P_signals[1,:][mask])
            D_prop_beta[i].extend(C.prop_P_signals[2,:][mask])
            D_prop_gamma[i].extend(C.prop_P_signals[-1,:][mask])
            D_alpha_delta[i].extend(alpha_delta[mask])
            D_beta_delta[i].extend(beta_delta[mask])
            D_gamma_delta[i].extend(gamma_delta[mask])
            D_beta_alpha[i].extend(beta_alpha[mask])
            D_gamma_alpha[i].extend(gamma_alpha[mask])
            D_gamma_beta[i].extend(gamma_beta[mask])
            D_hf_lf[i].extend(hf_lf[mask])
            D_50_q[i].extend(C.freqs_quantiles[0,:][mask])
            D_75_q[i].extend(C.freqs_quantiles[1,:][mask])
            D_85_q[i].extend(C.freqs_quantiles[2,:][mask])
            D_95_q[i].extend(C.freqs_quantiles[-1,:][mask])
            D_supp[i].extend(C.supp[mask])
            D_line_length[i].extend(C.line_length[mask])
            D_entropy[i].extend(C.entropy[mask])
            D_be[i].extend(C.be[mask])
            D_f_central[i].extend(C.f_central[mask])
         
    except:
        print(name, ' did not work')


#--- save
np.save('box_plot_data/prop_delta', D_prop_delta, allow_pickle=True)
np.save('box_plot_data/prop_alpha', D_prop_alpha, allow_pickle=True)
np.save('box_plot_data/prop_beta', D_prop_beta, allow_pickle=True)
np.save('box_plot_data/prop_gamma', D_prop_gamma, allow_pickle=True)
np.save('box_plot_data/alpha_delta', D_alpha_delta, allow_pickle=True)
np.save('box_plot_data/beta_delta', D_beta_delta, allow_pickle=True)
np.save('box_plot_data/gamma_delta', D_gamma_delta, allow_pickle=True)
np.save('box_plot_data/beta_alpha', D_beta_alpha, allow_pickle=True)
np.save('box_plot_data/gamma_alpha', D_gamma_alpha, allow_pickle=True)
np.save('box_plot_data/gamma_beta', D_gamma_beta, allow_pickle=True)
np.save('box_plot_data/hf_lf', D_hf_lf, allow_pickle=True)
np.save('box_plot_data/f_50_q', D_50_q, allow_pickle=True)
np.save('box_plot_data/f_75_q', D_50_q, allow_pickle=True)
np.save('box_plot_data/f_85_q', D_50_q, allow_pickle=True)
np.save('box_plot_data/f_95_q', D_50_q, allow_pickle=True)
np.save('box_plot_data/supp', D_supp, allow_pickle=True)
np.save('box_plot_data/line_length', D_line_length, allow_pickle=True)
np.save('box_plot_data/entropy', D_entropy, allow_pickle=True)
np.save('box_plot_data/be', D_be, allow_pickle=True)
np.save('box_plot_data/f_central', D_f_central, allow_pickle=True)