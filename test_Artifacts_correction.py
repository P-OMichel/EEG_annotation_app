import numpy as np
import matplotlib.pyplot as plt
from Functions.time_frequency import spectrogram
from Functions.detect_artifacts import find_artifacts
from Functions.WaveletQuantileNormalization import WQN_3

y = np.load('recordings_npy/rec_20240411_122350.npy')
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
except:  
    y_corr = y

t_spectro, f_spectro, spectro = spectrogram(y, fs)

t_spectro_corr, f_spectro_corr, spectro_corr = spectrogram(y_corr, fs)

fig, axes =plt.subplots(3, sharex = True)
axes[0].plot(t, y, color = 'black')
axes[0].plot(t, y_corr, color = 'red')
axes[1].pcolormesh(t_spectro, f_spectro, np.log(spectro + 0.0000001), shading = 'nearest', cmap = 'jet', vmin = np.log(0.001), vmax = np.log(20))
axes[2].pcolormesh(t_spectro_corr, f_spectro_corr, np.log(spectro_corr + 0.0000001), shading = 'nearest', cmap = 'jet', vmin = np.log(0.001), vmax = np.log(20))

plt.show()