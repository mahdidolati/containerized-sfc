import numpy as np
from constants import Const

class Vnf:
    def __init__(self, all_layers):
        self.cpu = np.random.randint(*Const.VNF_CPU)
        self.ram = np.random.randint(*Const.VNF_RAM)
        self.alpha = np.random.uniform(*Const.ALPHA_RANGE)
        layer_ids = np.random.choice(a=list(all_layers.keys()), size=np.random.randint(*Const.VNF_LAYER))
        self.layers = dict()
        for i in layer_ids:
            self.layers[i] = all_layers[i]


class Sfc:
    def __init__(self, t, vnfs):
        self.max_delay = 1.0
        self.traffic_rate = np.random.uniform(*Const.LAMBDA_RANGE)
        self.tau1 = t + np.random.randint(*Const.TAU1)
        self.tau2 = self.tau1 + np.random.randint(*Const.TAU2)
        self.vnfs = vnfs
        self.entry_point = None

    def vnf_in_rate(self, i):
        a = 1.0
        for j in range(i-1):
            a = a * self.vnfs[j].alpha
        return a * self.traffic_rate

    def cpu_req(self, i):
        return self.vnf_in_rate(i) * self.vnfs[i].cpu

    def ram_req(self, i):
        return self.vnf_in_rate(i) * self.vnfs[i].ram

    def __str__(self):
        return "max_delay: {}\nt1-t2: {}-{}".format(self.max_delay, self.tau1, self.tau2)

    def __repr__(self):
        return self.__str__()


class SfcGenerator:
    def __init__(self, my_net):
        self.my_net = my_net
        self.layers = dict()
        for i in range(Const.LAYER_NUM):
            self.layers[i] = np.random.randint(*Const.LAYER_SIZE)  # in megabytes
        self.vnfs = dict()
        self.vnf_num = Const.VNF_NUM
        for i in range(self.vnf_num):
            self.vnfs[i] = Vnf(self.layers)

    def get_chain(self, t):
        vnfs = list()
        n = np.random.randint(*Const.SFC_LEN)
        for _ in range(n):
            i = np.random.randint(0, self.vnf_num)
            vnfs.append(self.vnfs[i])
        new_sfc = Sfc(t, vnfs)
        new_sfc.entry_point = self.my_net.get_random_base_state()
        return new_sfc
