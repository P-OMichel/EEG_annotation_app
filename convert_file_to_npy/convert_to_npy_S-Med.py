import numpy as np
import json
import os

folder_path = 'C:/Users/holcman/Desktop/EEG data/St-Etienne'

for filename in os.listdir(folder_path):
    sub_folder_path = folder_path + '/' + filename
    try:
        for sub_filename in os.listdir(sub_folder_path):

            if '.json' in sub_filename:
                with open(folder_path + '/' + filename + '/' + sub_filename, 'r') as json_file:
                    data = json.load(json_file)
                y = np.array(data['EEGData'])
                np.save('recordings_npy/' + sub_filename[:-5] + '.npy', y)

    except:
        print(filename)



