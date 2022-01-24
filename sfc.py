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
    def __int__(self):
        self.vnf_id = 0
        self.cpu = np.random.uniform(*Const.VNF_CPU)
        self.ram = np.random.uniform(*Const.VNF_RAM)
        self.alpha = np.random.uniform(*Const.ALPHA_RANGE)
        self.layers = dict()

    def __init__(self, v_id, sharable_list, layer_cnt, layers, sharable_pr):
        self.vnf_id = v_id
        self.cpu = np.random.uniform(*Const.VNF_CPU)
        self.ram = np.random.uniform(*Const.VNF_RAM)
        self.alpha = np.random.uniform(*Const.ALPHA_RANGE)
        self.layers = dict()
        v_layer = np.random.randint(*Const.VNF_LAYER)
        s_layer = int(np.ceil(v_layer * sharable_pr))
        s_layer_pr = [(s_lid+1.0)/(sum(sharable_list) + len(sharable_list)) for s_lid in sharable_list]
        s_layer_list = np.random.choice(a=sharable_list, size=s_layer, p=s_layer_pr)
        for l_id in s_layer_list:
            self.layers[l_id] = layers[l_id]
        n_layer = int(np.floor(v_layer * (1.0 - sharable_pr)))
        for n_id in range(layer_cnt, layer_cnt+n_layer):
            self.layers[n_id] = np.random.randint(*Const.LAYER_SIZE)  # in megabytes

    def get_copy(self, new_layers):
        vnf = Vnf()
        vnf.vnf_id = self.vnf_id
        vnf.cpu = self.cpu
        vnf.ram = self.ram
        vnf.alpha = self.alpha
        self.layers = new_layers


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
        self.T1 = range(self.arrival_time, self.tau1)
        self.T2 = range(self.tau1, self.tau2 + 1)
        self.layers = dict()
        for i in range(len(self.vnfs)):
            for r in self.vnfs[i].layers:
                if r not in self.layers:
                    self.layers[r] = self.vnfs[i].layers[r]

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
        cr = self.vnf_in_rate(i) * self.vnfs[i].cpu
        # print("Cpu req is: {}".format(cr))
        return cr

    def ram_req(self, i):
        rr = self.vnf_in_rate(i) * self.vnfs[i].ram
        # print("Ram req is: {}".format(rr))
        return rr

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
            sharable_num = int(np.ceil(orgs[org] * Const.LAYER_NUM))
            sharable_list = list()
            for _ in range(sharable_num):
                sharable_list.append(layer_cnt)
                self.layers[layer_cnt] = np.random.randint(*Const.LAYER_SIZE)  # in megabytes
                layer_cnt += 1
            self.org_vnfs[org] = list()
            org_vnf_num = int(np.ceil(orgs[org] * Const.VNF_NUM))
            for i in range(org_vnf_num):
                a_vnf = Vnf(vnf_cnt, sharable_list, layer_cnt, self.layers, sharable_pr)
                vnf_cnt += 1
                self.org_vnfs[org].append(a_vnf)
                self.vnfs_list.append(a_vnf)
                for l_id in a_vnf.layers:
                    if l_id not in self.layers:
                        self.layers[l_id] = a_vnf.layers[l_id]
                        layer_cnt = layer_cnt + 1

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
