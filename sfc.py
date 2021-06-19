import numpy as np
from constants import Const


class LayerDownload:
    def __init__(self):
        self.download_data = dict()
        self.added_layers = set()

    def add_data(self, t, l, r):
        if t not in self.download_data:
            self.download_data[t] = list()
        l.add_dl(t, r)
        self.download_data[t].append((r, l))

    def cancel_download(self):
        for t in self.download_data:
            for r, l in self.download_data[t]:
                l.rm_dl(t, r)
        self.download_data = dict()


class Vnf:
    def __init__(self, layer_ids, all_layers, layer_pr=None):
        self.cpu = np.random.randint(*Const.VNF_CPU)
        self.ram = np.random.randint(*Const.VNF_RAM)
        self.alpha = np.random.uniform(*Const.ALPHA_RANGE)
        layer_ids = np.random.choice(a=layer_ids, size=np.random.randint(*Const.VNF_LAYER), p=layer_pr)
        self.layers = dict()
        for i in layer_ids:
            self.layers[i] = all_layers[i]


class Sfc:
    def __init__(self, t, vnfs):
        self.max_delay = np.random.uniform(*Const.SFC_DELAY)
        self.traffic_rate = np.random.uniform(*Const.LAMBDA_RANGE)
        self.arrival_time = t
        self.tau1 = t + np.random.randint(*Const.TAU1)
        self.tau2 = self.tau1 + np.random.randint(*Const.TAU2)
        self.vnfs = vnfs
        self.entry_point = None
        self.used_servers = set()

    def vnf_in_rate(self, i):
        a = 1.0
        for j in range(i-1):
            a = a * self.vnfs[j].alpha
        return a * self.traffic_rate

    def cpu_req(self, i):
        return self.vnf_in_rate(i) * self.vnfs[i].cpu

    def ram_req(self, i):
        return self.vnf_in_rate(i) * self.vnfs[i].ram

    def reset(self):
        self.used_servers = set()

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
        layer_pr = []
        for layer_no in self.layers:
            layer_pr.append(1.0 / (layer_no+1))
        s = 0
        for x in layer_pr:
            s += x
        for i in range(len(layer_pr)):
            layer_pr[i] = layer_pr[i] / s
        self.vnfs = dict()
        self.vnf_num = Const.VNF_NUM
        for i in range(self.vnf_num):
            self.vnfs[i] = Vnf(list(range(Const.LAYER_NUM)), self.layers, layer_pr)

    def get_chain(self, t):
        vnfs = list()
        n = np.random.randint(*Const.SFC_LEN)
        for _ in range(n):
            i = np.random.randint(0, self.vnf_num)
            vnfs.append(self.vnfs[i])
        new_sfc = Sfc(t, vnfs)
        new_sfc.entry_point = self.my_net.get_random_base_state()
        return new_sfc

    def get_download(self, t1, t2, r):
        new_sfc = Sfc(t1, [])
        new_sfc.tau1 = t1
        new_sfc.tau2 = t2
        new_sfc.traffic_rate = r
        return new_sfc
