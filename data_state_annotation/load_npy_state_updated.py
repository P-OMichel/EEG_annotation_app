import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path_data = 'data_state_annotation/D_rec_20240125_094414.npy'

print(path_data.split('/'))
my_file = Path(path_data)
if my_file.is_file():
    data = np.load(path_data, allow_pickle=True).item()


    plt.plot(data['t_list'], data['state'])
    plt.plot(data['t_list'], data['state_updated'])

    plt.show()

else: 
    print('file does not exist')