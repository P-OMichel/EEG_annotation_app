import numpy as np

# saved data for state update readable by the app
data = np.load('data_state_annotation/D_EEG_data_000063172.npy', allow_pickle=True)

print(data)

# dictionnary of features associated to states
data = np.load('box_plot_data/D.npy', allow_pickle=True).item()

i = 5
if type(i) == int:
    print('test')