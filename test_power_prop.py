import numpy as np
import matplotlib.pyplot as plt 
from Functions.time_frequency import spectrogram
from state_annotation.compute import Compute

#--- load data
y = np.load('recordings_npy/rec_20240321_085300.npy')
N = len(y)
fs = 128
t = np.linspace(0, N/fs, N)

#--- compute spectrogram
t_spectro, f_spectro, spectro = spectrogram(y, fs)

C = Compute()

C.get_data(t, y, fs, Ws = 30 *fs, step = 10 * fs)

C.run()

fig, axes = plt.subplots(3, sharex = True)
axes[0].pcolormesh(t_spectro, f_spectro, np.log(spectro + 0.0000001), shading = 'nearest', cmap = 'rainbow', vmin = np.log(0.001), vmax = np.log(20))
for i in range(4):
    axes[1].semilogy(C.t_list, C.P_signals[i,:])
    axes[2].plot(C.t_list, C.prop_P_signals[i,:])

plt.show()