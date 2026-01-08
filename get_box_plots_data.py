'''
This file iterates over each annotated recordings and computes the metrics. 

These metrics are stored in a dictionnary with the state as key and a list 
of the metrics values for this given state
'''

import numpy as np
import scipy as sc
import os
from state_annotation.compute import Compute
from Functions.detect_artifacts import find_artifacts
from Functions.WaveletQuantileNormalization import WQN_3

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
D = {}
D['D_prop_delta'] = {i: [] for i in range(22)}
D['D_prop_alpha'] = {i: [] for i in range(22)}
D['D_prop_beta'] = {i: [] for i in range(22)}
D['D_prop_gamma'] = {i: [] for i in range(22)}
D['D_alpha_delta'] = {i: [] for i in range(22)}
D['D_beta_delta'] = {i: [] for i in range(22)}
D['D_gamma_delta'] = {i: [] for i in range(22)}
D['D_beta_alpha'] = {i: [] for i in range(22)}
D['D_gamma_alpha'] = {i: [] for i in range(22)}
D['D_gamma_beta'] = {i: [] for i in range(22)}
D['D_hf_lf'] = {i: [] for i in range(22)}
D['D_50_q'] = {i: [] for i in range(22)}
D['D_75_q'] = {i: [] for i in range(22)}
D['D_85_q'] = {i: [] for i in range(22)}
D['D_95_q'] = {i: [] for i in range(22)}
D['D_supp'] = {i: [] for i in range(22)}
D['D_line_length'] = {i: [] for i in range(22)}
D['D_entropy'] = {i: [] for i in range(22)}
D['D_be'] = {i: [] for i in range(22)}
D['D_f_central'] = {i: [] for i in range(22)}


#--- iterate over each recordings
folder_path = 'data_state_annotation_07_01_2026'  
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

        list_detection = [2*fs, 1*fs, [0.0004,0.012], "sym4", 4, "periodization"] 
        list_WQN = ["sym4", "periodization", 30, 1]
    
        try:
            #--- detect artefacts
            # detection of the artifacts
            index_mask = find_artifacts(y,*list_detection)      

            # if it has some artifacts  
            if len(index_mask) != 0:                                              
            #--- clean the artifacts using WQN algorithm
                y_corr = WQN_3(y, index_mask, *list_WQN)  

                print('artifacts corrected')
                
        except:  
            y_corr = y

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
            D['D_prop_delta'][i].extend(C.prop_P_signals[0,:][mask])
            D['D_prop_alpha'][i].extend(C.prop_P_signals[1,:][mask])
            D['D_prop_beta'][i].extend(C.prop_P_signals[2,:][mask])
            D['D_prop_gamma'][i].extend(C.prop_P_signals[-1,:][mask])
            D['D_alpha_delta'][i].extend(alpha_delta[mask])
            D['D_beta_delta'][i].extend(beta_delta[mask])
            D['D_gamma_delta'][i].extend(gamma_delta[mask])
            D['D_beta_alpha'][i].extend(beta_alpha[mask])
            D['D_gamma_alpha'][i].extend(gamma_alpha[mask])
            D['D_gamma_beta'][i].extend(gamma_beta[mask])
            D['D_hf_lf'][i].extend(hf_lf[mask])
            D['D_50_q'][i].extend(C.freqs_quantiles[0,:][mask])
            D['D_75_q'][i].extend(C.freqs_quantiles[1,:][mask])
            D['D_85_q'][i].extend(C.freqs_quantiles[2,:][mask])
            D['D_95_q'][i].extend(C.freqs_quantiles[-1,:][mask])
            D['D_supp'][i].extend(C.supp[mask])
            D['D_line_length'][i].extend(C.line_length[mask])
            D['D_entropy'][i].extend(C.entropy[mask])
            D['D_be'][i].extend(C.be[mask])
            D['D_f_central'][i].extend(C.f_central[mask])
         
    except:
        print(name, ' did not work')


#--- save
np.save('box_plot_data_07_01_2026/D', D, allow_pickle=True)
