import numpy as np
from Functions.filter import get_filtered_signal
import Functions.sliding_fct as sliding
from Functions.compute_state import get_state_0_20

class Compute:

    def __init__(self):
        super().__init__()

    def get_data(self, t, y, fs, Ws, step):

        self.t = t
        self.y = y 
        self.fs = fs
        self.Ws = Ws
        self.step = step

    def get_power(self):

        signals = get_filtered_signal(self.y, self.fs, [[0.1,4],[7,14],[15,30],[30,45]])
        self.t_list, self.P_signals = sliding.power_nD(signals, self.t, self.Ws, self.step)

    def get_power_prop(self):

        self.prop_P_signals = self.P_signals / np.sum(self.P_signals, axis = 0)

    def get_supp_ratio(self):

        self.IES_prop, self.alpha_supp_prop = sliding.supp_power_prop(self.y, self.t, self.Ws, self.step, self.fs)[-2:]
        self.supp = self.alpha_supp_prop + 2 * self.IES_prop

    def get_state(self):

        N = len(self.t_list)
        self.state = np.zeros(N)
        for i in range(N):
            self.state[i] = get_state_0_20(self.supp[i], self.prop_P_signals[:, i])

    def run(self):

        self.get_power()
        self.get_power_prop()
        self.get_supp_ratio()
        self.get_state()