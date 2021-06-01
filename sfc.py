import numpy as np
from constants import Const


class Vnf:
    def __init__(self, all_layers):
        self.cpu = np.random.randint(*Const.VNF_CPU)
        self.ram = np.random.randint(*Const.VNF_RAM)
        self.alpha = np.random.uniform(*Const.ALPHA_RANGE)
        self.layers = np.random.choice(a=all_layers, size=np.random.randint(*Const.VNF_LAYER))


class Sfc:
    def __init__(self, t, vnfs):
        self.max_delay = 1.0
        self.traffic_rate = np.random.uniform(*Const.LAMBDA_RANGE)
        self.tau1 = t + np.random.randint(*Const.TAU1)
        self.tau2 = self.tau1 + np.random.randint(*Const.TAU2)
        self.vnfs = vnfs

    def __str__(self):
        return "max_delay: {}\nt1-t2: {}-{}".format(self.max_delay, self.tau1, self.tau2)


class SfcGenerator:
    def __init__(self):
        self.layers = dict()
        for i in range(Const.LAYER_NUM):
            self.layers[i] = np.random.randint(*Const.LAYER_SIZE)  # in megabytes
        self.vnfs = dict()
        self.vnf_num = Const.VNF_NUM
        for i in range(self.vnf_num):
            self.vnfs[i] = Vnf(list(self.layers.keys()))

    def get_chain(self, t):
        vnfs = list()
        n = np.random.randint(*Const.SFC_LEN)
        for _ in range(n):
            i = np.random.randint(0, self.vnf_num)
            vnfs.append(self.vnfs[i])
        return Sfc(t, vnfs)
