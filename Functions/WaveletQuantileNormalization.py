'''
-------------------------------------------------------------------------
   Artifacts removal using Wavelet Quantile Normalization
   Dora M., Holcman D., "Adaptive single-channel EEG artifact removal for
   real-time clinical monitoring", 2021
   
   Returns the artifact free EEG records.
-------------------------------------------------------------------------
''' 

import numpy as np
import pywt
from Functions.ecdf import *

#WQN with ecdf
#WQN_2 Matteo's method and percentage for selected ref and art coefficients parts
#WQN_3 Matteo's method


def WQN(y,index_mask,wavelet_name,mode,min_interval,alpha):
    '''
    input :
    - y              <-- the original non corrected eeg signal (1D list or array)
    - mask_artifacts <-- list of length=len(y) that has 1 at the positions where there is no artifacts and 0 where there is one 
    - index_mask     <-- marks the position of the iteration each time there is a change in 1/0 (list with a lenght twice as much as the number of artifacts)
    - mode           <-- mode to use for the DWT and IDWT (periodization, symmetric, reflect,... see pywt doc for more option choices)
    - min_interval   <-- size of the smallest desired interval at the last level of DWT 
    
    output :
    - post_WQN_signal <-- the cleaned eeg signal corrected thanks to the Wavelet Quantile Normalization algorithms (WQN)   
    '''

    post_WQN_signal=y.copy()                 # copy of the raw signal
    index_mask=[0]+index_mask+[len(y)-1]     # add starting point at 0 and final at the last index of y

    # iteration artifact by artifact 
    for Nb_art in range(1,len(index_mask)-1,2):             # len(index_mask) is always even and len(index_mask)/2 is the number of artifact
      
      # determine size of window to use given the min_interval
      size_art=index_mask[Nb_art+1]-index_mask[Nb_art]      # determine the size of the artifact

      max_level=int(np.log2(size_art/min_interval))         # max level decomposition for DWT given the minimum interval to reach on the last coefficient
      if max_level<1:                                       # if the max level is 0 the artifact is too small we skip to the next artifact (next iteration in the for loop)
         continue

      half_size_ref=max(min_interval*2**max_level,size_art)          # gives the max size of reference signal that can be taken to apply the same max_level on the DWT if available. 
      a=max(index_mask[Nb_art-1],index_mask[Nb_art]-half_size_ref)   # gives the beginning of the window on which DWT is applied
      b=min(index_mask[Nb_art+2],index_mask[Nb_art+1]+half_size_ref) # gives the ending of the window on which DWT is applied

      # compute the DWT on the window
      coeffs=pywt.wavedec(y[a:b],wavelet_name,mode,max_level)

      # we compute the percentages of the signal before and after, it should remain the same through DWT
      per_before=(index_mask[Nb_art]-a)/(b-a)  # percentage of respective ref signal before compared to total window for DWT
      per_after=(b-index_mask[Nb_art+1])/(b-a) # percentage of respective ref signal before compared to total window for DWT

      for coeff in coeffs:
         # determine the reference signal intervals and the artifact one in the coefficient
         coeff_size=coeff.size
         coeff_ref=[coeff[:int(coeff_size*per_before)],coeff[int(np.ceil(coeff_size*(1-per_after))):]] # take respective values for ref coefficients
         coeff_art=coeff[int(coeff_size*per_before):int(np.ceil(coeff_size*(1-per_after)))]
         if len(coeff_ref[0]) and len(coeff_ref[1])==0:
            continue
         coeff_ref=np.concatenate((coeff_ref[0],coeff_ref[1])) # transforms into a single list
         #ECDF of ref
         values_ref,CDF_ref=ecdf(np.abs(coeff_ref))  # values_ref is the list of all different values of coeff_ref
         #ECDF of art
         values_art,inv_art,CDF_art=ecdf_inv(np.abs(coeff_art))
         #T
         #T=interp_expand(CDF_art,CDF_ref,values_ref)
         T=np.interp(CDF_art,CDF_ref,values_ref)
         T=T[inv_art]
         #lambda
         T_norm=T/np.abs(coeff_art)

         coeff[int(coeff_size*per_before):int(np.ceil(coeff_size*(1-per_after)))]*=np.minimum(1,T_norm)**alpha
      
      #iDWT
      iDWt_values=pywt.waverec(coeffs,wavelet_name,mode)
      # it can lead to a signal that does not have the same size as b-a
      post_WQN_signal[a:b]=iDWt_values[:b-a]
   
    return post_WQN_signal

def WQN_2(y,index_mask,wavelet_name,mode,min_interval,alpha):
    '''
    input :
    - y              <-- the original non corrected eeg signal (1D list or array)
    - mask_artifacts <-- list of length=len(y) that has 1 at the positions where there is no artifacts and 0 where there is one 
    - index_mask     <-- marks the position of the iteration each time there is a change in 1/0 (list with a lenght twice as much as the number of artifacts)
    - mode           <-- mode to use for the DWT and IDWT (periodization, symmetric, reflect,... see pywt doc for more option choices)
    - min_interval   <-- size of the smallest desired interval at the last level of DWT 
    
    output :
    - post_WQN_signal <-- the cleaned eeg signal corrected thanks to the Wavelet Quantile Normalization algorithms (WQN)   
    '''
    post_WQN_signal=y.copy()                 # copy of the raw signal
    index_mask=[0]+index_mask+[len(y)-1]     # add starting point at 0 and final at the last index of y

   # iteration artifact by artifact 
    for Nb_art in range(1,len(index_mask)-1,2):             # len(index_mask) is always even and len(index_mask)/2 is the number of artifact
      
      # determine size of window to use given the min_interval
      size_art=index_mask[Nb_art+1]-index_mask[Nb_art]      # determine the size of the artifact

      max_level=int(np.log2(size_art/min_interval))         # max level decomposition for DWT given the minimum interval to reach on the last coefficient
      if max_level<1:                                       # if the max level is 0 the artifact is too small we skip to the next artifact (next iteration in the for loop)
         continue

      half_size_ref=max(min_interval*2**max_level,size_art)          # gives the max size of reference signal that can be taken to apply the same max_level on the DWT if available. 
      a=max(index_mask[Nb_art-1],index_mask[Nb_art]-half_size_ref)   # gives the beginning of the window on which DWT is applied
      b=min(index_mask[Nb_art+2],index_mask[Nb_art+1]+half_size_ref) # gives the ending of the window on which DWT is applied
      # compute the DWT on the window
      coeffs=pywt.wavedec(y[a:b],wavelet_name,mode,max_level)

      for coeff in coeffs:
         # determine the reference signal intervals and the artifact one in the coefficient
         # we compute the percentages of the signal before and after, it should remain the same through DWT
         per_before=(index_mask[Nb_art]-a)/(b-a)  # percentage of respective ref signal before compared to total window for DWT
         per_after=(b-index_mask[Nb_art+1])/(b-a) # percentage of respective ref signal before compared to total window for DWT
         coeff_size=coeff.size
         i=int(coeff_size*per_before)
         j=int(np.ceil(coeff_size*(1-per_after)))
         coeff_ref=[coeff[:i],coeff[j:]] # take respective values for ref coefficients
         coeff_art=coeff[i:j]
         if len(coeff_ref[0]) and len(coeff_ref[1])==0:
            continue
         
         # Transport the CDFs of the absolute value
         order = np.argsort(np.abs(coeff_art))
         inv_order = np.empty_like(order)
         inv_order[order] = np.arange(len(order))

         vals_ref = np.abs(np.concatenate(coeff_ref))
         ref_order = np.argsort(vals_ref)
         ref_sp = np.linspace(0, len(inv_order), len(ref_order))
         vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

         # Attenuate the coefficients
         r = vals_norm / np.abs(coeff[i:j])
         coeff[i:j] *= np.minimum(1, r) ** alpha   

      #iDWT
      iDWt_values=pywt.waverec(coeffs,wavelet_name,mode)
      # it can lead to a signal that does not have the same size as b-a
      post_WQN_signal[a:b]=iDWt_values[:b-a]
    
    return post_WQN_signal

def WQN_3(y,index_mask,wavelet_name,mode,min_interval,alpha):
    '''
    input :
    - y              <-- the original non corrected eeg signal (1D list or array)
    - mask_artifacts <-- list of length=len(y) that has 1 at the positions where there is no artifacts and 0 where there is one 
    - index_mask     <-- marks the position of the iteration each time there is a change in 1/0 (list with a lenght twice as much as the number of artifacts)
    - mode           <-- mode to use for the DWT and IDWT (periodization, symmetric, reflect,... see pywt doc for more option choices)
    - min_interval   <-- size of the smallest desired interval at the last level of DWT 
    
    output :
    - post_WQN_signal <-- the cleaned eeg signal corrected thanks to the Wavelet Quantile Normalization algorithms (WQN)   
    '''
    post_WQN_signal=y.copy()                 # copy of the raw signal
    index_mask=[0]+index_mask+[len(y)-1]     # add starting point at 0 and final at the last index of y

   # iteration artifact by artifact 
    for Nb_art in range(1,len(index_mask)-1,2):             # len(index_mask) is always even and len(index_mask)/2 is the number of artifact
      
      # determine size of window to use given the min_interval
      size_art=index_mask[Nb_art+1]-index_mask[Nb_art]      # determine the size of the artifact
      print(size_art)
      if size_art == 0:
         continue
      
      max_level=int(np.log2(size_art/min_interval))         # max level decomposition for DWT given the minimum interval to reach on the last coefficient
      if max_level<1:                                       # if the max level is 0 the artifact is too small we skip to the next artifact (next iteration in the for loop)
         continue

      half_size_ref=max(min_interval*2**max_level,size_art)          # gives the max size of reference signal that can be taken to apply the same max_level on the DWT if available. 
      a=max(index_mask[Nb_art-1],index_mask[Nb_art]-half_size_ref)   # gives the beginning of the window on which DWT is applied
      b=min(index_mask[Nb_art+2],index_mask[Nb_art+1]+half_size_ref) # gives the ending of the window on which DWT is applied
      # compute the DWT on the window
      coeffs=pywt.wavedec(y[a:b],wavelet_name,mode,max_level)

      # implement the WQN using different segmentation for DWT on ref and on art

      for coeff in coeffs:
         k = int(np.round(np.log2(b - a) - np.log2(coeff.size)))
         i, j = np.array([index_mask[Nb_art] - a, index_mask[Nb_art+1] - a]) // 2**k
         coeff_ref = [coeff[:i], coeff[j:]]
         coeff_ref=[coeff[:i],coeff[j:]] # take respective values for ref coefficients
         coeff_art=coeff[i:j]
         if len(coeff_ref[0])==0 and len(coeff_ref[1])==0:
            continue
         
         # Transport the CDFs of the absolute value
         order = np.argsort(np.abs(coeff_art))
         inv_order = np.empty_like(order)
         inv_order[order] = np.arange(len(order))

         vals_ref = np.abs(np.concatenate(coeff_ref))
         ref_order = np.argsort(vals_ref)
         ref_sp = np.linspace(0, len(inv_order), len(ref_order))
         vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

         # Attenuate the coefficients
         r = vals_norm / np.abs(coeff[i:j])
         coeff[i:j] *= np.minimum(1, r) ** alpha   

      #iDWT
      iDWt_values=pywt.waverec(coeffs,wavelet_name,mode)
      # it can lead to a signal that does not have the same size as b-a
      post_WQN_signal[a:b]=iDWt_values[:b-a]
    
    return post_WQN_signal
      
         



         

         
      
