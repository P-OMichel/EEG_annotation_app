'''
Compute the Empirical Cumulative Distribution like the ecdf function of matlab. (https://stackoverflow.com/questions/33345780/empirical-cdf-in-python-similiar-to-matlabs-one)
'''

import numpy as np


def ecdf(array):
    '''
    Inputs: 
    - array <-- 1D array on which is computed the CDF

    Outputs:
    - cumprob  <-- cumulative probability for a given value
    - values <-- list of the values from the signal
    '''

    L_values,counts=np.unique(array,return_counts=True) # takes the different values from the signal and counts how many times they appear (they are sorted)
    cumprob=np.cumsum(counts)/len(array)                # list where each component is the CDF for the value in L_values at the same index

    return L_values,cumprob

'''
Compute the inverse of the Empirical Cumulative Distribution 
'''
def ecdf_inv(array):
    '''
    Inputs: 
    - array <-- 1D array on which is computed the CDF

    Outputs:
    - cumprob  <-- cumulative probability for a given value
    - values <-- list of the values from the signal
    '''

    L_values,inv,counts=np.unique(array,return_inverse=True,return_counts=True) # takes the different values from the signal and counts how many times they appear (they are sorted)
    cumprob=np.cumsum(counts)/len(array)                # list where each component is the CDF for the value in L_values at the same index

    return L_values,inv,cumprob

'''
interpolation function without any border thresholding like the one from numpy
'''
def interp_expand(val,x,f_x):
    L=[]
    for i in val:
        L.append(np.interp(i,x,f_x,left=(f_x[1]-f_x[0])/(x[1]-x[0])*(i-x[0])+f_x[0],right=(f_x[-1]-f_x[-2])/(x[-1]-x[-2])*(i-x[-2])+f_x[0-2]))
    return L