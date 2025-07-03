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

C.get_data(t, y, fs, Ws = 30 *fs, step = 10 * fs, Ws_line_length = 2 * fs, step_line_length = 1 * fs)

C.run()

fig, axes = plt.subplots(6, sharex = True)
axes[0].pcolormesh(t_spectro, f_spectro, np.log(spectro + 0.0000001), shading = 'nearest', cmap = 'rainbow', vmin = np.log(0.001), vmax = np.log(20))
for i in range(4):
    axes[1].semilogy(C.t_list, C.P_signals[i,:])
    axes[2].plot(C.t_list, C.prop_P_signals[i,:])
axes[3].plot(C.t_list, C.be, label = 'be')
axes[3].plot(C.t_list, C.entropy, label = 'entropy')
axes[4].plot(C.t_line_length, C.line_length)
for i in range(4):
    axes[5].plot(C.t_list, C.freqs_quantiles[i,:])


axes[1].set_title('Power of different waves')
axes[2].set_title('Power proportion of different waves')
axes[3].set_title('Entropy and block entropy of signal')
axes[4].set_title('Line length')
axes[5].set_title('Frequencies of quantiles')

plt.show()