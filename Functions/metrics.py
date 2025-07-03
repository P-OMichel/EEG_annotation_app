import numpy as np
from scipy.signal import welch

#-------------------------------------------------------------------------------------------#
#                                      Regularity                                           #
#-------------------------------------------------------------------------------------------#

def line_length(y):

    return np.sum(np.abs(np.diff(y)))



def freqs_quantiles(signal, sampling_rate, quantiles, nperseg):
    """
    Compute frequencies associated to multiple quantiles of cumulative power of the PSD.

    Parameters:
        signal (1D array): Input signal.
        sampling_rate (float): Sampling frequency in Hz.
        quantiles (list of float): Quantiles (e.g., [0.25, 0.5, 0.75, 0.9]).
        nperseg (int or None): Segment length for Welch method.

    Returns:
        quantiles_freqs (dict): Mapping from quantile to frequency in Hz.
    """

    freqs, psd = welch(signal, fs=sampling_rate, nperseg=nperseg)
    cum_power = np.cumsum(psd)
    cum_power /= cum_power[-1]  # normalize to [0, 1]

    N_quantiles = len(quantiles)
    quantiles_freqs = np.zeros(N_quantiles)
    for i in range(N_quantiles):
        idx = np.searchsorted(cum_power, quantiles[i])
        quantile_freq = freqs[idx] if idx < len(freqs) else freqs[-1]
        quantiles_freqs[i] = quantile_freq

    return quantiles_freqs


