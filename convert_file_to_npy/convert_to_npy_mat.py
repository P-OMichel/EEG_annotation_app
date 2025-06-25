import numpy as np
import scipy as sc
import json
from scipy.signal import resample_poly

def resample_signal(signal, original_rate=250, target_rate=128):
    """
    Resample a 1D signal from original_rate to target_rate.
    
    Parameters:
    - signal: array-like, the input signal
    - original_rate: int, original sampling frequency in Hz (default 250)
    - target_rate: int, target sampling frequency in Hz (default 128)
    
    Returns:
    - resampled_signal: array-like, signal resampled to target_rate
    """
    gcd = np.gcd(original_rate, target_rate)
    up = target_rate // gcd
    down = original_rate // gcd
    
    resampled_signal = resample_poly(signal, up, down)
    return resampled_signal

def convert(name):
    '''
    file must be in recordings so the path is recordings/+filename.extension
    '''

    N_folder = len('recordings/')

    #--- load data
    if 'mat' in name:
        file_name = name[N_folder:-4]
        data = sc.io.loadmat(name)
        y = data['record'][:,0]
        fs = 128

    elif 'log' in name:
        file_name = name[N_folder:-4]
        file = open(name,'r')
        lines = file.readlines()
        N = len(lines)
        y = []
        for i in range(N):
            y.append(float(lines[i]))
        y = np.array(y)
        fs = 128
    
    elif 'npy' in name:
        file_name = name[N_folder:-4]
        y = np.load(name)
        fs = 128

    elif 'json' in name:
        file_name = name[N_folder:-5]
        with open(name, 'r') as file:
            data = json.load(file)            
        y = np.array(data['MNDRY_EEG_ELEC_POTL_BIS_TEMPR']['waveform'])  
        fs = 250  
        y = resample_signal(y, fs, 128)

    else:
        print('error when opening the file')


    #--- adjust EEG
    m_y = np.mean(y)
    if np.abs(m_y) > 10 :
        y = y - np.median(y)

    # save to npy 
    N_folder = len('recordings/')
    np.save('recordings_npy/' + file_name + '.npy', y)

    return y


#convert('recordings/rec_20240125_094414.mat')

import os

# Set the path to the folder you want to inspect
folder_path = 'recordings/'  # Replace with your folder path

# List all files and directories in the specified folder
for filename in os.listdir(folder_path):

    name = folder_path + filename

    convert(name)