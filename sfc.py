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
    def __init__(self, v_id, layer_ids, all_layers, n_share_p, layer_pr):
        self.vnf_id = v_id
        self.cpu = np.random.uniform(*Const.VNF_CPU)
        self.ram = np.random.uniform(*Const.VNF_RAM)
        self.alpha = np.random.uniform(*Const.ALPHA_RANGE)
        v_layer = np.random.randint(*Const.VNF_LAYER)
        s_layer = int(np.ceil(v_layer * n_share_p))
        layer_ids = np.random.choice(a=layer_ids, size=s_layer, p=layer_pr)
        self.layers = dict()
        for i in layer_ids:
            self.layers[i] = all_layers[i]
        n_layer = int(np.ceil(v_layer * (1-n_share_p)))
        new_layer_id = max(all_layers.keys()) + 1
        for i in range(n_layer):
            self.layers[i+new_layer_id] = np.random.randint(*Const.LAYER_SIZE)  # in megabytes


class Sfc:
    def __init__(self, t, vnfs):
        self.max_delay = np.random.randint(*Const.SFC_DELAY)
        # print(self.max_delay)
        self.traffic_rate = np.random.randint(*Const.LAMBDA_RANGE)
        self.arrival_time = t
        self.tau1 = t + np.random.randint(*Const.TAU1)
        self.tau2 = self.tau1 + np.random.randint(*Const.TAU2)
        self.vnfs = vnfs
        self.entry_point = None
        self.used_servers = set()

    def vnf_in_rate(self, i):
        a = 1.0
        for j in range(i):
            a = a * self.vnfs[j].alpha
        return a * self.traffic_rate

    def vnf_out_rate(self, i):
        a = 1.0
        for j in range(i):
            a = a * self.vnfs[j].alpha
        return a * self.traffic_rate * self.vnfs[i].alpha

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
    def __init__(self, my_net, n_share_p=1.0):
        self.my_net = my_net
        self.layers = dict()
        sharable_ids = list(range(Const.LAYER_NUM))
        for i in sharable_ids:
            self.layers[i] = np.random.randint(*Const.LAYER_SIZE)  # in megabytes
        self.no_share_id = Const.LAYER_NUM
        layer_pr = []
        for layer_id in sharable_ids:
            layer_pr.append(1.0 / (layer_id+1))
        s = 0
        for x in layer_pr:
            s += x
        for i in range(len(layer_pr)):
            layer_pr[i] = layer_pr[i] / s
        self.vnfs_list = list()
        self.vnf_num = Const.VNF_NUM
        for i in range(self.vnf_num):
            a_vnf = Vnf(i, sharable_ids, self.layers, n_share_p, layer_pr)
            for r in a_vnf.layers:
                if r not in self.layers:
                    self.layers[r] = a_vnf.layers[r]
            self.vnfs_list.append(a_vnf)
            # print("vnf layers: ", self.vnfs[i].layers.keys())

    def get_chain(self, t):
        vnfs = list()
        n = np.random.randint(*Const.SFC_LEN)
        for _ in range(n):
            i = np.random.randint(0, self.vnf_num)
            vnfs.append(self.vnfs_list[i])
        new_sfc = Sfc(t, vnfs)
        new_sfc.entry_point = self.my_net.get_random_base_state()
        return new_sfc

    def get_download(self, t1, t2, r):
        new_sfc = Sfc(t1, [])
        new_sfc.tau1 = t1
        new_sfc.tau2 = t2
        new_sfc.traffic_rate = r
        return new_sfc

    def print(self):
        print("Layers: {}".format(self.layers))
        for vnf in self.vnfs_list:
            print("Vnf: {} -> {}".format(vnf.vnf_id, vnf.layers))
