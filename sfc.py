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
    def __init__(self, v_id, sharable_list, nonsharable_list, layers, sharable_pr):
        self.vnf_id = v_id
        self.cpu = np.random.uniform(*Const.VNF_CPU)
        self.ram = np.random.uniform(*Const.VNF_RAM)
        self.alpha = np.random.uniform(*Const.ALPHA_RANGE)
        self.layers = dict()
        v_layer = np.random.randint(*Const.VNF_LAYER)
        s_layer = int(np.ceil(v_layer * sharable_pr))
        n_layer = int(np.floor(v_layer * (1.0 - sharable_pr)))
        s_layer_pr = [(s_lid+1.0)/(sum(sharable_list) + len(sharable_list)) for s_lid in sharable_list]
        s_layer_list = np.random.choice(a=sharable_list, size=s_layer, p=s_layer_pr)
        if len(nonsharable_list) > 0:
            n_layer_list = np.random.choice(a=nonsharable_list, size=n_layer)
        else:
            n_layer_list = list()
        for l_id in s_layer_list:
            self.layers[l_id] = layers[l_id]
        for l_id in n_layer_list:
            self.layers[l_id] = layers[l_id]


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

    def __str__(self):
        a = [v.vnf_id for v in self.vnfs]
        return "entry_point: {} --> {}\n\tmax_delay: {}, t1-t2: {}-{}".format(self.entry_point, a, self.max_delay, self.tau1, self.tau2)

    def __repr__(self):
        return self.__str__()


class SfcGenerator:
    def __init__(self, my_net, orgs, sharable_pr=1.0):
        self.my_net = my_net
        self.layers = dict()
        layer_cnt = 0
        vnf_cnt = 0
        self.orgs = orgs
        self.org_vnfs = dict()
        self.vnfs_list = list()
        self.vnf_num = Const.VNF_NUM
        for org in orgs:
            sharable_num = int(np.ceil(orgs[org] * Const.LAYER_NUM * sharable_pr))
            nonsharable_num = int(np.ceil(orgs[org] * Const.LAYER_NUM * (1.0 - sharable_pr)))
            sharable_list = list()
            nonsharable_list = list()
            for _ in range(sharable_num):
                sharable_list.append(layer_cnt)
                self.layers[layer_cnt] = np.random.randint(*Const.LAYER_SIZE)  # in megabytes
                layer_cnt += 1
            for _ in range(nonsharable_num):
                nonsharable_list.append(layer_cnt)
                self.layers[layer_cnt] = np.random.randint(*Const.LAYER_SIZE)  # in megabytes
                layer_cnt += 1
            self.org_vnfs[org] = list()
            org_vnf_num = int(np.ceil(orgs[org] * Const.VNF_NUM))
            for i in range(org_vnf_num):
                a_vnf = Vnf(vnf_cnt, sharable_list, nonsharable_list, self.layers, sharable_pr)
                vnf_cnt += 1
                self.org_vnfs[org].append(a_vnf)
                self.vnfs_list.append(a_vnf)
                for l_id in a_vnf.layers:
                    if l_id in nonsharable_list:
                        nonsharable_list.remove(l_id)

    def get_chain(self, t):
        org_list = [oo for oo in self.orgs]
        org_pr = [self.orgs[oo] for oo in org_list]
        org = np.random.choice(org_list, p=org_pr)
        n = np.random.randint(*Const.SFC_LEN)
        vnfs = np.random.choice(self.org_vnfs[org], size=min(n, len(self.org_vnfs[org])), replace=False)
        new_sfc = Sfc(t, vnfs)
        new_sfc.entry_point = self.my_net.get_random_base_state()
        return new_sfc

    def print(self):
        print("Layers: {}".format(self.layers))
        for vnf in self.vnfs_list:
            print("Vnf: {} -> {}".format(vnf.vnf_id, vnf.layers))
