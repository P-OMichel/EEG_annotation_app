import numpy as np
import scipy as sc
from scipy.ndimage import median_filter, gaussian_filter, uniform_filter, minimum_filter, maximum_filter
#-----------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------- time-frequency representation -------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def spectrogram(y, fs, delta_f=1, n_overlap=16, f_cut=45):

    # compute spectrogram
    nfft = int(fs / delta_f)
    overlap = nfft - n_overlap
    f_spectro, t_spectro, spectro = sc.signal.spectrogram(y, fs, nperseg=nfft, noverlap=overlap)
    delta_f = f_spectro[1] - f_spectro[0]
    j = int(f_cut / delta_f)
    f_spectro = f_spectro[:j]
    spectro = spectro[:j, :]  

    return t_spectro, f_spectro, spectro

#-----------------------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------------------------#
#------------------------- Edge frequency using the limit value on a time-frequency representation ---------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def edge_frequencies_limit_value(M, f, min_val = 0.001, max_val=20, threshold=None, T=5, q=0.75, threshold_min=0.01, threshold_max=10, factor_hf=5, smooth = False, second_check = True):
    '''
    M: time_frequency matrix
    f: associated frequencies
    min_val: minimum value to clip M
    max_val: maximum value to clip M 
    T: Value by which to multiply to get the threshold
    q: quantile value to get the on the M elements
    threshold_min: minimum value for the threshold
    threshold_max: maximum value for the threshold
    smooth: when True the edge_frequency is smoothed with a (3,1) Savitsky-Golay filter
    '''

    # mask of high frequency high power parts
    T_M = (M >= max_val/10).astype(int)
    count = np.sum(T_M[35:,:], axis=0)
    mask_hf = 1 - (count >= 1).astype(int)
    

    # thresholding   
    if threshold == None:
        if np.sum(mask_hf) != 0 and np.prod(mask_hf) !=1:
            threshold = T * np.quantile(np.clip(M[:, mask_hf == 1], min_val, max_val), q)
        else:
            threshold = T * np.quantile(np.clip(M, min_val, max_val), q)
        # min max values 
        threshold = min(threshold_max, threshold)
        threshold = max(threshold_min, threshold)

    # get list of frequencies:
    edge_frequencies = get_edge_limit_value(M, f, threshold)

    # smooth
    if smooth:
        edge_frequencies = sc.signal.savgol_filter(edge_frequencies,3,1)

    # estimation of the hf edge frequency
    threshold_hf = threshold / factor_hf

    # get list of frequencies:
    edge_frequencies_hf = get_edge_limit_value(M, f, threshold_hf)

    # smooth
    if smooth:
        edge_frequencies_hf = sc.signal.savgol_filter(edge_frequencies_hf,3,1)
    

    return edge_frequencies, edge_frequencies_hf, threshold


def get_edge_limit_value(spectro, f_spectro, threshold):
    '''
    Inputs:
    spectro: clipped spectrogram  
    N_max: maximum number of point per colums higher than the threshold

    Ouput:
    '''

    # cumulative of each colums
    reversed_spectro = np.flipud(spectro)
    reversed_cumulative_spectro = np.cumsum(reversed_spectro, axis=0)
    cumulative_spectro = np.flipud(reversed_cumulative_spectro)

    cond = cumulative_spectro <= threshold
    indices = np.argmax(cond, axis = 0)
    # check where the condition cannot be reached as it return 0 and change to -1 to get highest frequency later
    valid = cond.any(axis=0)
    indices[~valid] = -1

    edge_frequencies = f_spectro[indices]

    return edge_frequencies

#-----------------------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------------------------#
#--------------------- Edge frequency using the significant values on a time-frequency representation ------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def edge_frequencies_significant_value(M, f, min_val=0.001, max_val=20, threshold=None, T=5, q=0.5, threshold_min=0.05, threshold_max=2, N_max=2, smooth = False):
    
    # thresholding
    if threshold == None:
        threshold = T * np.quantile(np.clip(M, min_val, max_val), q)
        # min max values 
        threshold = min(threshold_max, threshold)
        threshold = max(threshold_min, threshold)

    # thresholding on the TF matrix
    thresholded_M = np.zeros_like(M)
    thresholded_M[np.where(M >= threshold)] = 1

    # get list of frequencies:
    edge_frequencies = get_edge_significant_value(thresholded_M, f, N_max)

    # smooth
    if smooth:
        edge_frequencies = sc.signal.savgol_filter(edge_frequencies,3,1)

    return edge_frequencies, threshold

def edge_frequencies_significant_value_hf(M, f, min_val=0.001, max_val=20, threshold=None, T=5, q=0.5, threshold_min=0.05, threshold_max=2, N_max=2, factor_hf = 10, smooth = False, second_check = True):
    
    # thresholding
    if threshold == None:
        threshold = T * np.quantile(np.clip(M, min_val, max_val), q)
        # min max values 
        threshold = min(threshold_max, threshold)
        threshold = max(threshold_min, threshold)

    # thresholding on the TF matrix
    thresholded_M = np.zeros_like(M)
    thresholded_M[np.where(M >= threshold)] = 1

    # get list of frequencies:
    edge_frequencies = get_edge_significant_value(thresholded_M, f, N_max)

    # smooth
    if smooth:
        edge_frequencies = sc.signal.savgol_filter(edge_frequencies,3,1)

    # call function again after removing IES potential parts

    if threshold == None and second_check:
        mask = (edge_frequencies >= 4).astype(int)
        threshold = T * np.quantile(np.clip(M[:, mask == 1], min_val, max_val), q)
        # min max values 
        threshold = min(threshold_max, threshold)
        threshold = max(threshold_min, threshold)

        thresholded_M = np.zeros_like(M)
        thresholded_M[np.where(M >= threshold)] = 1

        # get list of frequencies:
        edge_frequencies = get_edge_significant_value(thresholded_M, f, N_max)

        # smooth
        if smooth:
            edge_frequencies = sc.signal.savgol_filter(edge_frequencies,3,1)

    # estimation of the hf edge frequency
    threshold_hf = threshold / factor_hf

    thresholded_M_hf = np.zeros_like(M)
    thresholded_M_hf[np.where(M >= threshold_hf)] = 1

    # get list of frequencies:
    edge_frequencies_hf = get_edge_significant_value(thresholded_M, f, N_max)

    # smooth
    if smooth:
        edge_frequencies_hf = sc.signal.savgol_filter(edge_frequencies_hf,3,1)
    

    return edge_frequencies, edge_frequencies_hf, threshold

def get_edge_significant_value(thresholded_M, f, N_max):
    '''
    Inputs:
    spectro: clipped spectrogram  
    N_max: maximum number of point per colums higher than the threshold

    Ouput:
    '''
    # create matrix where in a column 0,0--> 0 | 0,1 --> 0 | 1,0 --> 0 | 1,1 --> 1 
    M_01 = thresholded_M[:-1, :] * thresholded_M[1:, :]

    # cumulative of each columns of the thresholded matrix
    reversed_spectro = np.flipud(thresholded_M)
    reversed_cumulative_spectro = np.cumsum(reversed_spectro, axis=0)
    cumulative_spectro = np.flipud(reversed_cumulative_spectro)

    # find first frequency where M_01 is 1 or when the cumulative reaches N_max
    N = len(thresholded_M[0,:])
    edge_frequencies = np.zeros(N)
    for j in range(N):
        try:
            i = max(np.where(cumulative_spectro[:, j] == N_max)[0])
            k = max(np.where(M_01[:, j] == 1)[0]) + 1
            index = max(i,k)
            edge_frequencies[j] = f[index] # check if it cannot be higher than index in f_spectro
        except:
            edge_frequencies[j] = 0 # already 0 check if can be changed

    return edge_frequencies #, cumulative_spectro


#-----------------------------------------------------------------------------------------------------------------------#
#------------------------------------ Threshold spectrogram per frequency band -----------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def combined_FTTFM(M, f, 
                   list_freq = [1, 4, 7, 14, 20, 30], 
                   T = [0.2]*7,#[1,    1,   1,   1,   1,   1,   1], 
                   q = [0.9]*7):#[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]):

    delta_f = f[1] - f[0]

    #--- first frequency
    j = int((list_freq[0] - f[0]) / delta_f)
    M[:j, :] = threshold_M(M[:j, :], T[0], q[0])

    #--- last_frequency
    i = int((list_freq[-1] - f[0]) / delta_f)
    M[i:, :] = threshold_M(M[i:, :], T[-1], q[-1])

    #--- subdivisions in between
    for k in range(len(list_freq) - 1):
        i = int((list_freq[k] - f[0]) / delta_f) 
        j = int((list_freq[k + 1] - f[0]) / delta_f) 
        M[i:j,:] = threshold_M(M[i:j,:], T[k + 1], q[k + 1])

    return f, M

def FTTFM(M, f, f_int, T, q):
    '''
    Frequency Thresholded Time-Frequency Matrix
    '''

    delta_f = f[1] - f[0]
    i = int((f_int[0] - f[0]) / delta_f)
    j = int((f_int[-1] - f[0]) / delta_f)
    M = M[i : j, :]

    M = threshold_M(M, T, q)

    return M
    
def threshold_M(M, T, q):

    M = (M >= T * np.quantile(M, q)).astype(int)
    
    return M