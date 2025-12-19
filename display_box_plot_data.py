import numpy as np
import matplotlib.pyplot as plt

prop_delta = np.load('prop_delta.npy', allow_pickle=True).item()
prop_delta = [prop_delta[i] for i in range(22)]
prop_alpha = np.load('prop_alpha.npy', allow_pickle=True).item()
prop_alpha = [prop_alpha[i] for i in range(22)]
prop_beta = np.load('prop_beta.npy', allow_pickle=True).item()
prop_beta  = [prop_beta [i] for i in range(22)]
prop_gamma = np.load('prop_gamma.npy', allow_pickle=True).item()
prop_gamma = [prop_gamma[i] for i in range(22)]

alpha_delta = np.load('alpha_delta.npy', allow_pickle=True).item()
alpha_delta = [alpha_delta[i] for i in range(22)]
beta_delta = np.load('beta_delta.npy', allow_pickle=True).item()
beta_delta = [beta_delta[i] for i in range(22)]
gamma_delta = np.load('gamma_delta.npy', allow_pickle=True).item()
gamma_delta = [gamma_delta[i] for i in range(22)]
beta_alpha = np.load('beta_alpha.npy', allow_pickle=True).item()
beta_alpha = [beta_alpha[i] for i in range(22)]
gamma_alpha = np.load('gamma_alpha.npy', allow_pickle=True).item()
gamma_alpha = [gamma_alpha[i] for i in range(22)]
gamma_beta = np.load('gamma_beta.npy', allow_pickle=True).item()
gamma_beta = [gamma_alpha[i] for i in range(22)]
hf_lf = np.load('hf_lf.npy', allow_pickle=True).item()
hf_lf = [hf_lf[i] for i in range(22)]

f_50_q = np.load('f_50_q.npy', allow_pickle=True).item()
f_50_q = [f_50_q[i] for i in range(22)]
f_75_q = np.load('f_75_q.npy', allow_pickle=True).item()
f_75_q = [f_75_q[i] for i in range(22)]
f_85_q = np.load('f_85_q.npy', allow_pickle=True).item()
f_85_q = [f_85_q[i] for i in range(22)]
f_95_q = np.load('f_95_q.npy', allow_pickle=True).item()
f_95_q = [f_95_q[i] for i in range(22)]

supp = np.load('supp.npy', allow_pickle=True).item()
supp = [supp[i] for i in range(22)]

line_length = np.load('line_length.npy', allow_pickle=True).item()
line_length = [line_length[i] for i in range(22)]

entropy = np.load('entropy.npy', allow_pickle=True).item()
entropy = [entropy[i] for i in range(22)]

be = np.load('be.npy', allow_pickle=True).item()
be = [be[i] for i in range(22)]

fig, axes = plt.subplots(4)

axes[0].boxplot(prop_delta, positions=range(22), patch_artist=True)
axes[0].set_title('prop_delta')
axes[1].boxplot(prop_alpha, positions=range(22), patch_artist=True)
axes[1].set_title('prop_alpha')
axes[2].boxplot(prop_beta, positions=range(22), patch_artist=True)
axes[2].set_title('prop_beta')
axes[3].boxplot(prop_gamma, positions=range(22), patch_artist=True)
axes[3].set_title('prop_gamma')

for i in range(4):
    axes[i].set_ylim(0,1)

fig, axes = plt.subplots(7)

axes[0].boxplot(alpha_delta, positions=range(22), patch_artist=True)
axes[0].set_title('alpha_delta')
axes[1].boxplot(beta_delta, positions=range(22), patch_artist=True)
axes[1].set_title('beta_delta')
axes[2].boxplot(gamma_delta, positions=range(22), patch_artist=True)
axes[2].set_title('gamma_delta')
axes[3].boxplot(beta_alpha, positions=range(22), patch_artist=True)
axes[3].set_title('beta_alpha')
axes[4].boxplot(gamma_alpha, positions=range(22), patch_artist=True)
axes[4].set_title('gamma_alpha')
axes[5].boxplot(gamma_beta, positions=range(22), patch_artist=True)
axes[5].set_title('gamma_beta')
axes[6].boxplot(hf_lf, positions=range(22), patch_artist=True)
axes[6].set_title('hf_lf')

for i in range(7):
    axes[i].set_ylim(0,1)

fig, axes = plt.subplots(4)

axes[0].boxplot(f_50_q, positions=range(22), patch_artist=True)
axes[0].set_title('f_50_q')
axes[1].boxplot(f_75_q, positions=range(22), patch_artist=True)
axes[1].set_title('f_75_q')
axes[2].boxplot(f_50_q, positions=range(22), patch_artist=True)
axes[2].set_title('f_85_q')
axes[3].boxplot(f_95_q, positions=range(22), patch_artist=True)
axes[3].set_title('f_95_q')

for i in range(4):
    axes[i].set_ylim(0,20)

fig, axes = plt.subplots(4)

axes[0].boxplot(supp, positions=range(22), patch_artist=True)
axes[0].set_title('supp')
axes[1].boxplot(line_length, positions=range(22), patch_artist=True)
axes[1].set_title('line_length')
axes[1].set_ylim(0,5)
axes[2].boxplot(entropy, positions=range(22), patch_artist=True)
axes[2].set_title('entropy')
axes[3].boxplot(be, positions=range(22), patch_artist=True)
axes[3].set_title('be')

plt.show()