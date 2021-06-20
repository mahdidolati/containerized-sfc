import numpy as np


class Solver:
    def __init__(self, my_net, layer_del_th):
        self.my_net = my_net
        self.layer_del_th = layer_del_th

    def usable_node(self, s, c, chain_req, i, t, delay_budget):
        for tt in range(chain_req.tau1, chain_req.tau2 + 1):
            if self.my_net.g.nodes[c]["nd"].cpu_avail(tt) < chain_req.cpu_req(i) or \
                    self.my_net.g.nodes[c]["nd"].ram_avail(tt) < chain_req.ram_req(i):
                return False, 0
        if c[0] == "b":
            return False, 0
        R, d = self.my_net.get_missing_layers(c, chain_req, i, chain_req.tau1)
        for tt in range(t, chain_req.tau2 + 1):
            if self.my_net.g.nodes[c]["nd"].disk_avail(tt) < d:
                return False, 0
        dl_result, dl_obj = self.my_net.do_layer_dl_test(c, R, d, t, chain_req.tau1-1)
        if not dl_result:
            return False, 0
        dl_obj.cancel_download()
        if s != c:
            path_bw, path_delay, links = self.my_net.get_biggest_path(s, c, chain_req.tau1, delay_budget)
            if path_bw == 0 or path_delay > delay_budget:
                return False, 0
            min_bw = self.my_net.get_min_bw(links, chain_req.tau1, chain_req.tau2)
            if min_bw < chain_req.vnf_in_rate(i):
                return False, 0
        return True, len(R)

    def cloud_embed(self, chain_req):
        prev = chain_req.entry_point
        delay_budge = chain_req.max_delay
        path_bw, path_delay, links = self.my_net.get_biggest_path(prev, "c", chain_req.tau1, delay_budge)
        if path_bw == 0 or path_delay > delay_budge:
            return False
        min_bw = self.my_net.get_min_bw(links, chain_req.tau1, chain_req.tau2)
        if min_bw < chain_req.vnf_in_rate(0):
            return False
        else:
            for l in links:
                l.embed(chain_req, 0)
        return True

    def solve(self, chain_req, t, sr):
        if self.cloud_embed(chain_req):
            return True, 0
        layer_download_vol = 0
        prev = chain_req.entry_point
        delay_budge = chain_req.max_delay
        active_dls = []
        for i in range(len(chain_req.vnfs)):
            cur_budge = delay_budge / (len(chain_req.vnfs) - i)
            N1 = self.my_net.get_random_edge_nodes(sr)
            C = dict()
            for c in N1:
                n_usable, r_needed = self.usable_node(prev, c, chain_req, i, t, cur_budge)
                if n_usable:
                    C[c] = r_needed
            if len(C) == 0:
                self.my_net.evict_sfc(chain_req)
                chain_req.reset()
                for a in active_dls:
                    a.cancel_download()
                return False, 0
            m = None
            br_needed = np.infty
            for mm in C:
                if C[mm] < br_needed:
                    m = mm
                    br_needed = C[mm]
            self.my_net.g.nodes[m]["nd"].embed(chain_req, i)
            chain_req.used_servers.add(m)
            R, d = self.my_net.get_missing_layers(m, chain_req, i, chain_req.tau1)
            layer_download_vol = layer_download_vol + d
            if len(R) > 0:
                dl_result, dl_obj = self.my_net.do_layer_dl_test(m, R, d, t, chain_req.tau1 - 1)
                active_dls.append(dl_obj)
                if self.my_net.share_layer:
                    self.my_net.g.nodes[m]["nd"].add_layer(R, chain_req)
                else:
                    self.my_net.g.nodes[m]["nd"].add_layer_no_share(R, chain_req)
            if prev != m:
                path_bw, path_delay, links = self.my_net.get_biggest_path(prev, m, chain_req.tau1, cur_budge)
                delay_budge = delay_budge - path_delay
                for l in links:
                    l.embed(chain_req, i)
            prev = m
        for m in chain_req.used_servers:
            for l in self.my_net.g.nodes[m]["nd"].layers:
                self.my_net.g.nodes[m]["nd"].layers[l].finalized = True
        return True, layer_download_vol

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

    def reset(self):
        self.my_net.reset()
        self.my_net.share_layer = False


class ShareSolver(Solver):
    def __init__(self, my_net, layer_del_th):
        super().__init__(my_net, layer_del_th)
        self.my_net.share_layer = True

    def get_name(self):
        return "Share-{}".format(self.layer_del_th)

    def reset(self):
        self.my_net.reset()
        self.my_net.share_layer = True