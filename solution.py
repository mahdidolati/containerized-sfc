import numpy as np


class Solver:
    def __init__(self, my_net, layer_del_th):
        self.my_net = my_net
        self.layer_del_th = layer_del_th

    def usable_node(self, s, c, chain_req, i, t, delay_budget):
        if c[0] == "b":
            return False
        if self.my_net.g.nodes[c]["nd"].cpu_avail(t) < chain_req.cpu_req(i):
            return False
        if self.my_net.g.nodes[c]["nd"].ram_avail(t) < chain_req.ram_req(i):
            return False
        R, d = self.my_net.get_missing_layers(c, chain_req, i, chain_req.tau1)
        if self.my_net.g.nodes[c]["nd"].disk_avail() < d:
            return False
        dl_result, dl_obj = self.my_net.do_layer_dl_test(c, R, d, t, chain_req.tau1-1)
        if not dl_result:
            return False
        if s != c:
            path_bw, path_delay, links = self.my_net.get_biggest_path(s, c, t, delay_budget)
            dl_obj.cancel_download()
            if path_bw < chain_req.vnf_in_rate(i):
                return False
        else:
            dl_obj.cancel_download()
        return True

    def solve(self, chain_req, t, sr):
        prev = chain_req.entry_point
        delay_budge = chain_req.max_delay
        active_dls = []
        node_new_layer = dict()
        for i in range(len(chain_req.vnfs)):
            cur_budge = delay_budge / (len(chain_req.vnfs) - i)
            N1 = self.my_net.get_random_edge_nodes(sr)
            N1 = np.append(N1, "c")
            C = list()
            for c in N1:
                if self.usable_node(prev, c, chain_req, i, t, cur_budge):
                    C.append(c)
            if len(C) == 0:
                for a in active_dls:
                    a.cancel_download()
                return False
            m = np.random.choice(C)
            self.my_net.g.nodes[m]["nd"].embed(chain_req, i)
            R, d = self.my_net.get_missing_layers(m, chain_req, i, chain_req.tau1)
            node_new_layer[m] = R
            dl_result, dl_obj = self.my_net.do_layer_dl_test(m, R, d, t, chain_req.tau1 - 1)
            active_dls.append(dl_obj)
            path_bw, path_delay, links = self.my_net.get_biggest_path(prev, m, t, cur_budge)
            delay_budge = delay_budge - path_delay
            for l in links:
                l.embed(chain_req, i)
            prev = m
        for m in node_new_layer:
            chain_req.used_servers.add(m)
            if self.my_net.share_layer:
                self.my_net.g.nodes[m]["nd"].add_layer(node_new_layer[m], chain_req)
            else:
                self.my_net.g.nodes[m]["nd"].add_layer_no_share(node_new_layer[m], chain_req)
        return True

    def handle_sfc_eviction(self, chain_req):
        self.my_net.evict_sfc(chain_req)
        for m in chain_req.used_servers:
            for l in self.my_net.g.nodes[m]["nd"].layers:
                self.my_net.g.nodes[m]["nd"].layers[l].remove_user(chain_req)
        chain_req.used_servers = set()

    def pre_arrival_procedure(self, t):
        to_be_delete = set()
        for m in self.my_net.g.nodes():
            for l in self.my_net.g.nodes[m]["nd"].layers:
                if len(self.my_net.g.nodes[m]["nd"].layers[l].chain_users) == 0:
                    if t - self.my_net.g.nodes[m]["nd"].layers[l].last_used > self.layer_del_th:
                        to_be_delete.add((m, l))
        for m, l in to_be_delete:
            del self.my_net.g.nodes[m]["nd"].layers[l]


class NoShareSolver(Solver):
    def __init__(self, my_net, layer_del_th):
        super().__init__(my_net, layer_del_th)
        self.my_net.share_layer = False

    def get_name(self):
        return "No Share-{}".format(self.layer_del_th)


class ShareSolver(Solver):
    def __init__(self, my_net, layer_del_th):
        super().__init__(my_net, layer_del_th)

    def get_name(self):
        return "Share-{}".format(self.layer_del_th)