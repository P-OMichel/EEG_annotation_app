'''
Functions computed on sliding windows with a specific window size and step in between two consecutive windows
'''
import numpy as np
from Functions.suppressions import detect_suppressions_power


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












