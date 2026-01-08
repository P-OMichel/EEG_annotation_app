'''
Detects the artifacts in the EEG signal using thresholds on wavelet coefficients CDF slopes.
ref. Jiaqi Wang master thesis: "Real-Time EEG artifact detection for continuous Monitoring during anesthesia"
'''
'''
minimize the loss of clean eeg using the slopes (which size of window to use ans step according to the size of an artifact to avoid computing slopes with a too large mix of clean and artifacted signal)
see if sliding mean on abs value of the signal can be good for detection
'''

import numpy as np
import pywt         # https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html for list of wavelet family name

from Functions.ecdf import ecdf
from scipy import stats


def is_outlier_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

def remove_short_art(signal,pos_art,ws,smoothing_ws):
    '''
    ws: number of neighbours to take on the left and right
    smoothing_ws: number of points to take for smoothing (associated to a cutting frequency)
    '''

    N=len(signal)
    fit = np.zeros(N)
    for pos in pos_art:

        i = pos[0] - ws
        j = pos[-1] + ws

        if i < 0:
            i = 0
        if j >=N:
            j = N - 1

        #signal[i:j] = signal[i:j] - sc.signal.savgol_filter(signal[i:j],smoothing_ws,1)
        fit[i:j] = sc.signal.savgol_filter(signal[i:j],smoothing_ws,1)

        # add polynomial fit
        x = np.array([i , j])
        y = np.array([signal[i], signal[j]])

        # Fit a quadratic (parabola) curve to these two points
        coeff = np.polyfit(x, y, 2)
        poly = np.poly1d(coeff)

        # Use the polynomial to estimate the missing value at index 'i'
        x_interp = range(i,j)
        fit[i:j] = fit[i:j] + poly(x_interp)  

    return fit

def find_artifacts(y,
                   Ws,
                   step,
                   threshold,
                   wavelet_name,
                   level,
                   mode):
    '''
    Inputs:
    - y              <-- raw eeg signal
    - Ws             <-- sliding window size on wich the DWD is applied
    - step           <-- step size between windows
    - threshold      <-- threshold for the slopes (list of 2, one for the approximation coefficient slope and the other for the first detail coefficient slope)
    - wavelet name   <-- name of the wavelet family to use
    - level          <-- number of detail coefficients
    - mode           <-- mode to use for DWT
    
    Outputs:
    - mask_artifacts <-- 1 at the position where there is no artifacts and 0 where there is one
    - index_mask     <-- marks the position of the iteration each time there is a change in 1/0
    '''

    N=len(y)      # length of the EEG signal that is sent as input (should be the same as the batch of data gathered from the device before doing compution an ideally should be a multiple of Ws)

    # gathering the slopes, thresholding and creating mask

    ## initialization
    start_window=0                              # marker of the position of the left window edge for the next computation

    # creating the mask    
    threshold_a,threshold_d1=threshold          # receives the thresholds value
    mask_artifacts=[1 for i in range(N)]        # initialize the mask
    mask_pos=[]                                 # set iteration indexes of where the artefacts start and end. (useful for Clean_EEG.py to avoid an unecessary search index in a list to get positions of 0 in the mask) 
    
    while start_window<=N:
        if start_window+Ws<=N:                           # checking that the marker for next computation + the size of the window is within the length of the signal

            s_a,s_d1=CDF_Slope(y[start_window:start_window+Ws],wavelet_name,level,mode)
            #med_amp=np.quantile(np.abs(y[start_window:start_window+Ws]),1)
        
            if (s_a<threshold_a or s_d1<threshold_d1):# or med_amp>=90) :  # slope lower than threshold_a indicating an artifact EOG or Motion and lower than threshold_d1 indicating an artifact EMG
                mask_artifacts[start_window:start_window+Ws]=[0 for i in range(Ws)]
                mask_pos.append([start_window,start_window+Ws]) # in case
            start_window+=step                           # will go above n in case of equality and stops the cycle

        elif (start_window<N and start_window+Ws>N):     # still within the signal but next window will have a part in the signal and another outside. si on prend plus de signal (au moin step en iteration de plus) ne pose plus de pblm (voir avec pour le spectro)
            
            s_a,s_d1=CDF_Slope(y[start_window:],wavelet_name,level,mode)

            if (s_a<threshold_a or s_d1<threshold_d1): # or med_amp>=90):    
                mask_artifacts[start_window:]=[0 for i in range(N-start_window)]
                mask_pos.append([start_window,N-1]) # in case
            start_window+=Ws                             # will be higher than N and this stops the cycle
    

    if len(mask_pos)<1:
        #print('no artifact detected')
        return []
    else:
        index_mask=[mask_pos[0][0]]
    for i in range(len(mask_pos)-1):
        if np.abs(mask_pos[i][1]-mask_pos[i+1][0])>step:
            index_mask.append(mask_pos[i][1])
            index_mask.append(mask_pos[i+1][0])
    index_mask.append(mask_pos[-1][1])

    return index_mask

def find_artifacts_mask(y,
                   Ws,
                   step,
                   threshold,
                   wavelet_name,
                   level,
                   mode):
    '''
    Inputs:
    - y              <-- raw eeg signal
    - Ws             <-- sliding window size on wich the DWD is applied
    - step           <-- step size between windows
    - threshold      <-- threshold for the slopes (list of 2, one for the approximation coefficient slope and the other for the first detail coefficient slope)
    - wavelet name   <-- name of the wavelet family to use
    - level          <-- number of detail coefficients
    - mode           <-- mode to use for DWT
    
    Outputs:
    - mask_artifacts <-- 1 at the position where there is no artifacts and 0 where there is one
    - index_mask     <-- marks the position of the iteration each time there is a change in 1/0
    '''

    N=len(y)      # length of the EEG signal that is sent as input (should be the same as the batch of data gathered from the device before doing compution an ideally should be a multiple of Ws)

    # gathering the slopes, thresholding and creating mask

    ## initialization
    start_window=0                              # marker of the position of the left window edge for the next computation

    # creating the mask    
    threshold_a,threshold_d1=threshold          # receives the thresholds value
    mask_artifacts=[1 for i in range(N)]        # initialize the mask
    mask_pos=[]                                 # set iteration indexes of where the artefacts start and end. (useful for Clean_EEG.py to avoid an unecessary search index in a list to get positions of 0 in the mask) 
    
    while start_window<=N:
        if start_window+Ws<=N:                           # checking that the marker for next computation + the size of the window is within the length of the signal

            s_a,s_d1=CDF_Slope(y[start_window:start_window+Ws],wavelet_name,level,mode)
            #med_amp=np.quantile(np.abs(y[start_window:start_window+Ws]),1)
        
            if (s_a<threshold_a or s_d1<threshold_d1):# or med_amp>=90) :  # slope lower than threshold_a indicating an artifact EOG or Motion and lower than threshold_d1 indicating an artifact EMG
                mask_artifacts[start_window:start_window+Ws]=[0 for i in range(Ws)]
                mask_pos.append([start_window,start_window+Ws]) # in case
            start_window+=step                           # will go above n in case of equality and stops the cycle

        elif (start_window<N and start_window+Ws>N):     # still within the signal but next window will have a part in the signal and another outside. si on prend plus de signal (au moin step en iteration de plus) ne pose plus de pblm (voir avec pour le spectro)
            
            s_a,s_d1=CDF_Slope(y[start_window:],wavelet_name,level,mode)

            if (s_a<threshold_a or s_d1<threshold_d1): # or med_amp>=90):    
                mask_artifacts[start_window:]=[0 for i in range(N-start_window)]
                mask_pos.append([start_window,N-1]) # in case
            start_window+=Ws                             # will be higher than N and this stops the cycle
    
        print(s_a, s_d1, start_window)
    return mask_artifacts

def CDF_Slope(y,wavelet_name,level,mode):
    '''
    Inputs: 
    - y            <-- part of the eeg signal on which the DWD is applied
    - wavelet name <-- name of the wavelet family to use
    - level        <-- number of detail coefficients
    - mode         <-- mode to use for DWT

    Outputs:
    - CDFa         <-- CDF of the approximation coefficient (if wanted)
    - CDFd1        <-- CDF of the first detail coefficient  (if wanted)
    - slope_a      <-- Slope joining the first and last point of CFDa 
    - slope_d1     <-- Slope joining the first and last point of CDFd1
    '''

    # Compute the first detail coefficient and the approximation one (using DWD)
    ca,cd1=coeffs_wavelet(y,wavelet_name,level,mode)

    # Compute the empirical CDF for each coefficient
    CDFa=ecdf(np.abs(ca)) # CDFa[0] is the list of values appearing in ca and CDFa[1] is the list of associated cumulative probability
    CDFd1=ecdf(np.abs(cd1))

    # Compute the slopes for each CDF using (ymax-ymin)/(xmax-xmin) adapted to the CDF 
    slope_a=(CDFa[1][-1]-CDFa[1][0])/(CDFa[0][-1]-CDFa[0][0])
    slope_d1=(CDFd1[1][-1]-CDFd1[1][0])/(CDFd1[0][-1]-CDFd1[0][0])

    return slope_a,slope_d1 

def coeffs_wavelet(y,wavelet_name,level,mode):
    '''
    Inputs:
    - y            <-- signal on which the DWD is applied
    - wavelet name <-- name of the wavelet family to use
    - level        <-- number of detail coefficients
    - mode         <-- mode to use for DWt (str)

    Outputs:
    - ca           <-- approximation coefficient array
    - cd1          <-- first detail coefficient array
    '''

    coeffs=pywt.wavedec(y,wavelet_name,mode,level) # discrete wavelet decomposition from the pywt library, it returns the coefficients 
    ca=coeffs[0]                                   # such as the approximation one is the first of the list and the first detail one the last
    cd1=coeffs[len(coeffs)-1]                 

    return ca,cd1

