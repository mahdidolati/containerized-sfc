import heapq
from opt_gurobi_single import solve_single
from relax_gurobi_single import solve_single_relax
from opt_ilp import get_ilp

class Solver:
    def __init__(self, my_net):
        self.my_net = my_net
        self.batch = False

    def sort_nodes_disk(self, all_nodes, chain_req, i):
        h = []
        counter = 0
        for n in all_nodes:
            R, d = self.my_net.get_missing_layers(n, chain_req, i, chain_req.tau1)
            heapq.heappush(h, (d, counter, n))
            counter = counter + 1
        return h

    def solve_batch(self, my_net, vnfs_list, R_ids, R_vols, reqs):
        pass

    def usable_node(self, s, c, chain_req, i, t, delay_budget):
        if c[0] == "b":
            return False, 0
        for tt in range(chain_req.tau1, chain_req.tau2 + 1):
            if self.my_net.g.nodes[c]["nd"].cpu_avail(tt) < chain_req.cpu_req(i) or \
                    self.my_net.g.nodes[c]["nd"].ram_avail(tt) < chain_req.ram_req(i):
                return False, 0
        R, d = self.my_net.get_missing_layers(c, chain_req, i, chain_req.tau1)
        for tt in range(chain_req.tau1, chain_req.tau2 + 1):
            if self.my_net.g.nodes[c]["nd"].disk_avail(tt) < d:
                return False, 0
        dl_result, dl_obj = self.my_net.do_layer_dl_test(c, R, d, t, chain_req.tau1-1)
        if not dl_result:
            return False, 0
        dl_obj.cancel_download()
        if s != c:
            path_bw, path_delay, path_nodes, links = self.my_net.get_biggest_path(s, c, chain_req.tau1, delay_budget)
            if path_bw == 0 or path_delay > delay_budget:
                return False, 0
            min_bw = self.my_net.get_min_bw(links, chain_req.tau1, chain_req.tau2)
            if min_bw < chain_req.vnf_in_rate(i):
                return False, 0
        return True, len(R)

    def cloud_embed(self, chain_req):
        prev = chain_req.entry_point
        delay_budge = chain_req.max_delay
        path_bw, path_delay, path_nodes, links = self.my_net.get_biggest_path(prev, "c", chain_req.tau1, delay_budge)
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
        mark_del_layer = dict()
        for i in range(len(chain_req.vnfs)):
            cur_budge = delay_budge / (len(chain_req.vnfs) - i)
            N1 = self.my_net.get_random_edge_nodes(sr)
            sorted_nodes = self.sort_nodes_disk(N1, chain_req, i)
            m = None
            while len(sorted_nodes) > 0:
                dl_need, cnt, c_node = heapq.heappop(sorted_nodes)
                n_usable, r_needed = self.usable_node(prev, c_node, chain_req, i, t, cur_budge)
                if n_usable:
                    m = c_node
                    break
            if m is None:
                self.my_net.evict_sfc(chain_req)
                chain_req.reset()
                for a in active_dls:
                    a.cancel_download()
                return False, 0
            self.my_net.g.nodes[m]["nd"].embed(chain_req, i)
            self.my_net.g.nodes[m]["nd"].mark_needed(chain_req, i)
            chain_req.used_servers.add(m)
            R, d = self.my_net.get_missing_layers(m, chain_req, i, chain_req.tau1)
            max_del = 0
            for tt in range(chain_req.tau1, chain_req.tau2 + 1):
                max_del = max(max_del, abs(self.my_net.g.nodes[m]["nd"].disk_avail(tt) - d))
            # this if can be true only in storage aware alg
            if max_del > 0:
                if m not in mark_del_layer:
                    mark_del_layer[m] = set()
                mark_del_layer[m].update(self.my_net.g.nodes[m]["nd"].get_unused_for_del(max_del))
            layer_download_vol = layer_download_vol + d
            if len(R) > 0:
                dl_result, dl_obj = self.my_net.do_layer_dl_test(m, R, d, t, chain_req.tau1 - 1)
                active_dls.append(dl_obj)
                self.my_net.g.nodes[m]["nd"].add_layer(R, chain_req)
            if prev != m:
                path_bw, path_delay, path_nodes, links = self.my_net.get_biggest_path(prev, m, chain_req.tau1, cur_budge)
                delay_budge = delay_budge - path_delay
                for l in links:
                    l.embed(chain_req, i)
            prev = m
        for m in chain_req.used_servers:
            for l in self.my_net.g.nodes[m]["nd"].layers:
                self.my_net.g.nodes[m]["nd"].layers[l].finalized = True
        for m in mark_del_layer:
            for l in mark_del_layer[m]:
                del self.my_net.g.nodes[m]["nd"].layers[l]
        return True, layer_download_vol

    def handle_sfc_eviction(self, chain_req, t):
        # print("-------------- handle eviction --------------------")
        self.my_net.evict_sfc(chain_req)
        # for m in chain_req.used_servers:
            # print("node {}: has capacity {} and availabe {} available-no-cache {}, has unused {}".format(m,
            #         self.my_net.g.nodes[m]["nd"].disk,
            #         self.my_net.g.nodes[m]["nd"].disk_avail(t),
            #         self.my_net.g.nodes[m]["nd"].disk_avail_no_cache(t),
            #         self.my_net.g.nodes[m]["nd"].has_unused_layer(t)))
        chain_req.used_servers = set()
        # print("---------------------------------------------------")

    def pre_arrival_procedure(self, t):
        pass

    def post_arrival_procedure(self, status, t, chain_req):
        pass


class NoShareSolver(Solver):
    def __init__(self, my_net, layer_del_th):
        super().__init__(my_net)
        self.layer_del_th = layer_del_th
        self.my_net.disable_layer_sharing()

    def get_name(self):
        return "NS-{}".format(self.layer_del_th)

    def reset(self):
        self.my_net.reset()
        self.my_net.disable_layer_sharing()


class ShareSolver(Solver):
    def __init__(self, my_net, layer_del_th):
        super().__init__(my_net)
        self.layer_del_th = layer_del_th
        self.my_net.enable_layer_sharing()

    def get_name(self):
        return "S-{}".format(self.layer_del_th)

    def reset(self):
        self.my_net.reset()
        self.my_net.enable_layer_sharing()


class GurobiBatch(Solver):
    def __init__(self, my_net, R_ids, R_vols):
        super().__init__(my_net)
        self.R_ids = R_ids
        self.R_vols = R_vols
        self.my_net.enable_layer_sharing()
        self.batch = True

    def get_name(self):
        return "GrBt"

    def solve_batch(self, my_net, vnfs_list, R_ids, R_vols, reqs):
        return get_ilp(reqs, self.my_net, self.R_ids, self.R_vols)

    def reset(self):
        self.my_net.reset()
        self.my_net.enable_layer_sharing()


class GurobiSingle(Solver):
    def __init__(self, my_net, R_ids, R_vols):
        super().__init__(my_net)
        self.R_ids = R_ids
        self.R_vols = R_vols
        self.my_net.enable_layer_sharing()

    def get_name(self):
        return "GrSi"

    def solve(self, chain_req, t, sr):
        return solve_single(self.my_net, self.R_ids, self.R_vols, chain_req)

    def reset(self):
        self.my_net.reset()
        self.my_net.enable_layer_sharing()


class GurobiSingleRelax(Solver):
    def __init__(self, my_net, R_ids, R_vols, eviction_strategy="default"):
        super().__init__(my_net)
        self.R_ids = R_ids
        self.R_vols = R_vols
        self.my_net.enable_layer_sharing()
        self.eviction_strategy = eviction_strategy

    def get_name(self):
        return "GrSiRlx(" + self.eviction_strategy[0] + ")"

    def solve(self, chain_req, t, sr):
        return solve_single_relax(self.my_net, self.R_ids, self.R_vols, chain_req)

    def pre_arrival_procedure(self, t):
        # print("-------------- pre arrival --------------------")
        # pre_state = []
        for m in self.my_net.g.nodes():
            if m[0] == "e":
                self.my_net.g.nodes[m]["nd"].make_s1()
                # print("node {}: has capacity {} and availabe {} available-no-cache {}, has unused {}".format(m,
                #                                                     self.my_net.g.nodes[m]["nd"].disk,
                #                                                     self.my_net.g.nodes[m]["nd"].disk_avail(t),
                #                                                     self.my_net.g.nodes[m]["nd"].disk_avail_no_cache(t),
                #                                                     self.my_net.g.nodes[m]["nd"].has_unused_layer(t)))
        # print("------------------------------------------------")

    def post_arrival_procedure(self, status, t, chain_req):
        # print("-------------- post arrival --------------------")
        for m in self.my_net.g.nodes():
            if m[0] == "e":
                # print("Node {}, storage: {}, {}".format(m,
                #                                         self.my_net.g.nodes[m]["nd"].disk_avail(t),
                #                                         self.my_net.g.nodes[m]["nd"].disk_avail_no_cache(t)))
                # Transition of admitting the req
                if self.eviction_strategy == "q_learning":
                    self.my_net.g.nodes[m]["nd"].make_s2()
                    self.my_net.g.nodes[m]["nd"].q_agent.add_transition(
                        self.my_net.g.nodes[m]["nd"].s1,
                        self.my_net.g.nodes[m]["nd"].get_local_kept(),
                        self.my_net.g.nodes[m]["nd"].get_local_reused(),
                        self.my_net.g.nodes[m]["nd"].s2
                    )
                elif self.eviction_strategy == "popularity_learn":
                    inuse = self.my_net.g.nodes[m]["nd"].get_all_inuse()
                    self.my_net.g.nodes[m]["nd"].p_agent.add_inuse(inuse)
                # Transition of emptying disk
                if self.my_net.g.nodes[m]["nd"].disk_avail(t) < 0:
                    # print("From {}: delete {}, unused {}".format(m, over_used, vol))
                    if self.eviction_strategy == "q_learning":
                        self.my_net.g.nodes[m]["nd"].make_s1()
                        self.my_net.g.nodes[m]["nd"].empty_storage(t)
                        self.my_net.g.nodes[m]["nd"].make_s2()
                        self.my_net.g.nodes[m]["nd"].q_agent.add_transition(
                            self.my_net.g.nodes[m]["nd"].s1,
                            self.my_net.g.nodes[m]["nd"].get_local_kept(),
                            self.my_net.g.nodes[m]["nd"].get_local_reused(),
                            self.my_net.g.nodes[m]["nd"].s2
                        )
                    elif self.eviction_strategy == "popularity_learn":
                        self.my_net.g.nodes[m]["nd"].empty_storage_popularity(t)
                    else:
                        self.my_net.g.nodes[m]["nd"].empty_storage_random(t)
                #

        # print("------------------------------------------------")

    def reset(self):
        self.my_net.reset()
        self.my_net.enable_layer_sharing()