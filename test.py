import matplotlib.pyplot as plt
import numpy as np


y = np.load('recordings_npy/rec_20240125_094414.npy')

res = plt.specgram(y, Fs = 128, NFFT=128, noverlap=128 - 16, cmap = 'rainbow', vmax = 1)
spec = res[0]
print(np.quantile(spec, 0.01))
print(np.quantile(spec, 0.99))
plt.show()
print(res)