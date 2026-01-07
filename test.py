import numpy as np

# # saved data for state update readable by the app
# data = np.load('data_state_annotation/D_EEG_data_000063172.npy', allow_pickle=True)

# print(data)

# # dictionnary of features associated to states
# data = np.load('box_plot_data/D.npy', allow_pickle=True).item()

# i = 5
# if type(i) == int:
#     print('test')


M = np.array([[5,1,2,6],
              [8,2,3,6],
              [9,5,3,5]])


a = list(range(4))
print(a)
b = [1,3]

common = set(a) & set(b)

a_clean = [x for x in a if x not in common]
print(a, a_clean)

M = M[:, a_clean]

print(M)