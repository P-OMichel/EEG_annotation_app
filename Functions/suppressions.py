import numpy as np
import scipy as sc
from Functions.filter import filter_butterworth
from Functions.utils import detect_pos_1, diff_envelops, envelope_maxima

def erosion_dilation(mask,min_band,max_gap,fs):
    '''
    Function that return a mask after erosion/dilatation/erosion

    Inputs:
    - mask      <-- mask of 1 at a position of a value of interest and 0 elsewhere
    - min_band  <-- min lenght of correct values (minimal lenght of segment of 1)
    - max_gap   <-- max lenght of incorrect values (maximal lenght of segment of 0)
    
    Outputs:
    - new_mask  <-- mask created after a process of erosion/dilatation/erosion (explain each phase)
    '''

    # translate the min_band and max_gap from time duration to interval length
    min_band = int(fs*min_band) 
    max_gap = int(fs*max_gap) - 1   # minus 1 to have the correct effect on erosion (erosion with int 3 for instance takes away 2 by construction of the function)

    # routine to erode, dilate and erode the mask
    new_mask = sc.ndimage.binary_erosion(mask,np.ones(min_band),iterations=1)
    new_mask = sc.ndimage.binary_dilation(new_mask,np.ones(min_band+max_gap),iterations=1)
    new_mask = sc.ndimage.binary_erosion(new_mask,np.ones(max_gap),iterations=1)

    return new_mask

def detect_suppressions_power(y, fs, T_IES_max = 12, T_alpha_max=5):
    ''''
    Function to detect the alpha-suppressions and IES
    '''
    
    N_points = int(fs/4)
    h = np.ones(N_points) / N_points #  0.25 s 

    # smoothed power of signal between [1.5,30]] Hz
    y2 = np.convolve(filter_butterworth(y,fs,[1.5,30])**2,h,mode='same')
    N = len(y2)
    
    # smoothed power of signal between [7,14]] Hz
    y2_alpha = np.convolve(filter_butterworth(y,fs,[7,14])**2,h,mode='same')

    # smoothed power of signal between [15,20] Hz
    y2_beta = np.convolve(filter_butterworth(y,fs,[15,20])**2,h,mode='same')

    # smoothed power of signal between [30,45]] Hz
    y2_gamma = np.convolve(filter_butterworth(y,fs,[40,45])**2,h,mode='same')

    #--- IES threshold
    # find if zone is ok or mostly suppression
    q = np.quantile(y2,0.75)
    
    if q <= 8: # condition indicating it is mostly suppresions
        T_IES = min(10,q*3)

    else:
        # take only values that are lower than very high values from high power region and artefact, ground checks
        # what happens if power in Burst is same as high values signal ?
        try:
            indices=np.where(y2<T_IES_max*12)[0]
            T_IES=np.quantile(y2[indices],0.9)*0.12
        except: # indices is empty
            T_IES=np.quantile(y2,0.9)*0.12

        # Threshold for IES
        T_IES = min(T_IES,T_IES_max)

    # if T_IES<=8: # can be if just a burst or artefact that make the quantile 75 high and 0.12*quantile(0.9) low #8
    #     print('T_IES', T_IES)
    #     T_IES = 8

    #--- alpha-suppressions threshold
    try:
        indices=np.where(y2_alpha<T_alpha_max*15)[0]
        T_alpha=np.quantile(y2_alpha[indices],0.9)*0.15
    except: # indices is empty
        T_alpha=np.quantile(y2_alpha,0.9)*0.15

        # threshold for alpha supp
        T_alpha = min(T_alpha, T_IES_max)

    # if T_alpha<=1: # can be if just a burst or artefact that make the quantile 75 high and 0.12*quantile(0.9) low #8
    #     print('T_alpha', T_alpha)
    #     T_IES = 1

    #--- beta threshold
    T_beta = T_alpha * 0.75

    #--- get shallow signals mask
    r_gamma_delta = np.convolve(filter_butterworth(y,fs,[30,45])**2,h,mode='same') / np.convolve(filter_butterworth(y,fs,[1,4])**2,np.ones(fs)/fs,mode='same')
    P_y = np.convolve(filter_butterworth(y,fs,[0.1,45])**2,h,mode='same')
    mask_shallow_signal = np.zeros(N)
    mask_shallow_signal[np.where((r_gamma_delta >= 0.05) & (P_y <= 100))[0]] = 1
    mask_shallow_signal = erosion_dilation(mask_shallow_signal,0.5,0.5,fs)*1

    #--- ground check mask
    mask_ground_check = get_mask_ground_check(y,fs)

    #--- mask of suppressions
    Om_alpha = np.where((y2_alpha < T_alpha) & (y2_beta < T_beta) & (mask_shallow_signal != 1) & (y2_gamma < 0.25) & (mask_ground_check != 1))[0]    # list of indices where the condition is satisfied for alpha-suppressions
    mask_alpha = np.zeros(N)
    mask_alpha[Om_alpha] = 1           #  set 1 where there is an alpha suppression, 0 elsewhere

    Om_IES = np.where((mask_alpha == 1) & (y2 < T_IES))[0]     # list of indices where the condition is satisfied for IES
    Om_IES = np.where((y2 < T_IES))
    mask_IES = np.zeros(N)
    mask_IES[Om_IES] = 1           #  set 1 where there is an IES suppression, 0 elsewhere

    #--- Erosion and dilatation routine
    mask_alpha = erosion_dilation(mask_alpha,0.6,0.5,fs)*1
    mask_IES = erosion_dilation(mask_IES,1.1,0.9,fs)*1 

    # remove alpha_supp where there is an IES
    mask_alpha[np.where((mask_alpha-mask_IES) != 1)[0]] = 0
    #mask_alpha = erosion_dilation(mask_alpha,0.5,0.5,fs)

    # get proportion of shallow signals
    shallow_signal_proportion = np.sum(mask_shallow_signal)/N

    # get position of suppressions
    pos_IES = detect_pos_1(mask_IES)
    pos_alpha = detect_pos_1(mask_alpha)

    # get proportion of IES in window
    IES_proportion = np.sum(mask_IES)/N
    alpha_suppression_proportion = np.sum(mask_alpha)/N

    return y2,y2_alpha,pos_IES,pos_alpha,shallow_signal_proportion, mask_IES, mask_alpha, IES_proportion,alpha_suppression_proportion

def get_mask_ground_check(y,fs, delta_f = 1, n_overlap = 16):

    # compute the spectrogram
    nfft = int(fs / delta_f)
    overlap = nfft - n_overlap
    f_spectro, t_spectro, spectro = sc.signal.spectrogram(y, fs, nperseg=nfft, noverlap=overlap)
    delta_f = f_spectro[1] - f_spectro[0]
    j = int(45 / delta_f)
    f_spectro = f_spectro[:j]
    spectro = spectro[:j, :]  

    # get sum of values above T_h in a spectro per colum
    T_h = 10
    T_h_spectro = np.zeros_like(spectro)
    T_h_spectro[np.where(spectro >= T_h)] = 1

    sum_h = np.sum(T_h_spectro, axis = 0)

    # get sum of values bellow T_l
    T_l = 0.005

    T_l_spectro = np.zeros_like(spectro)
    T_l_spectro[np.where(spectro <= T_l)] = 1

    sum_l = np.sum(T_l_spectro, axis = 0) 

    # mask of values higher or lower than threhsolds
    mask_h = np.zeros_like(sum_h)
    mask_h[np.where(sum_h >= 30)] = 1

    mask_l = np.zeros_like(sum_l)
    mask_l[np.where(sum_l >= 20)] = 1

    delta_t = t_spectro[1] - t_spectro[0]

    return delta_t, mask_l, mask_h   