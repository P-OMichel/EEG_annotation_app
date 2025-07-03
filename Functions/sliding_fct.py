'''
Functions computed on sliding windows with a specific window size and step in between two consecutive windows
'''
import numpy as np
from Functions.suppressions import detect_suppressions_power
from scipy.stats import entropy
from collections import Counter
from Functions.metrics import line_length, freqs_quantiles


#-------------------------------------------------------------------------------------------#
#                                 Power of signal                                           #
#-------------------------------------------------------------------------------------------#

def power_1D(signal,t,Ws,step):
    '''
    Output: 
    mean_power_list: 1D numpy array
    '''
    # create a list of all the sliding windows
    windows=[signal[i:i+Ws]**2 for i in range(0,len(signal)-Ws,step)]
    # compute the mean power in each window
    mean_power_list=np.array([np.median(win) for win in windows])
    # associated time list (time bin is for the end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,mean_power_list

def power_nD(signals,t,Ws,step):
    '''
    Input:
    signals: nD numpy array
    Output:
    mean_power_list: numpy array of same number of line as signals
    '''
    # create a list of all the sliding windows
    windows=[signals[:,i:i+Ws]**2 for i in range(0,np.shape(signals)[1]-Ws,step)]
    # compute the mean power in each window for axis 1
    mean_power_list=np.transpose([np.median(win,axis=1) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list, mean_power_list


#-------------------------------------------------------------------------------------------#
#                                 Fragmentation                                           #
#-------------------------------------------------------------------------------------------#

def supp_power(y,Ws,step,fs,T_IES_max,T_alpha_max):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    pos_IES, pos_alpha_supp = [], []
    for i in range(len(windows)):
        IES,alpha_supp=detect_suppressions_power(windows[i],fs,T_IES_max,T_alpha_max)[2:4]
        IES=[[pos[0]+i*step,pos[-1]+i*step] for pos in IES]
        alpha_supp=[[pos[0]+i*step,pos[-1]+i*step] for pos in alpha_supp]
        pos_IES+=IES
        pos_alpha_supp+=alpha_supp

    return pos_IES, pos_alpha_supp

def supp_power_prop(y,t,Ws,step,fs):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    IES_prop, alpha_supp_prop = [], []
    for i in range(len(windows)):
        IES, alpha_supp = detect_suppressions_power(windows[i],fs)[-2:]
        IES_prop.append(IES)
        alpha_supp_prop.append(alpha_supp)

    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list, np.array(IES_prop), np.array(alpha_supp_prop)


#-------------------------------------------------------------------------------------------#
#                                      Entropy                                              #
#-------------------------------------------------------------------------------------------#

def compute_entropy(signal, t, window_size, step, n_bins=10, normalize=True):
    entropies = []
    for i in range(0, len(signal) - window_size + 1, step):
        window = signal[i:i+window_size]
        if normalize:
            window = (window - np.mean(window)) / np.std(window)  # z-score
        # Discretize
        hist, _ = np.histogram(window, bins=n_bins, density=True)
        hist = hist[hist > 0]  # remove 0 entries
        ent = entropy(hist, base=2)
        entropies.append(ent)

    t_list=[t[window_size+i*step] for i in range(len(entropies))]

    return t_list, np.array(entropies)

def compute_block_entropy_k(signal, t, window_size, step, k=2, n_bins=10, normalize=True):
    entropies = []
    for i in range(0, len(signal) - window_size + 1, step):
        window = signal[i:i+window_size]
        if normalize:
            window = (window - np.mean(window)) / (np.std(window) + 1e-8)
        
        # Discretize
        quantized = np.digitize(window, np.histogram_bin_edges(window, bins=n_bins))
        
        # Create k-grams
        kgrams = [tuple(quantized[j:j+k]) for j in range(len(quantized) - k + 1)]
        freqs = Counter(kgrams)
        total = sum(freqs.values())
        probs = np.array([count / total for count in freqs.values()])
        
        # Shannon entropy of the k-grams
        ent = -np.sum(probs * np.log2(probs))
        entropies.append(ent)

    t_list=[t[window_size+i*step] for i in range(len(entropies))]
    
    return t_list, np.array(entropies)

#-------------------------------------------------------------------------------------------#
#                                      Regularity                                           #
#-------------------------------------------------------------------------------------------#

def compute_line_length(signal, t, Ws, step):
    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute the line length in each window
    line_length_list=np.array([line_length(win) for win in windows])
    # associated time list (time bin is for the end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list, line_length_list


#-------------------------------------------------------------------------------------------#
#                                   frequency quantile                                      #
#-------------------------------------------------------------------------------------------#

def compute_freqs_quantiles(signal, t, Ws, step, sampling_rate, quantiles=[0.5, 0.75, 0.85, 0.95], nperseg=None):
    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute the line length in each window
    freqs_list=np.array([freqs_quantiles(win, sampling_rate, quantiles, nperseg) for win in windows])
    # associated time list (time bin is for the end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list, np.transpose(freqs_list)











