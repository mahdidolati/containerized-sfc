import heapq
from opt_gurobi_single import solve_single
from relax_gurobi_single import RelaxSingle
from opt_ilp import solve_batch_opt
from sfc import LayerDownload
from test import TestResult
import numpy as np


class Solver:
    def __init__(self):
        self.batch = False
        self.convert_layer = False

    def get_name(self):
        pass

    def set_env(self, my_net, R_ids, R_vols):
        self.my_net = my_net
        self.my_net.enable_layer_sharing()
        self.R_ids = R_ids
        self.R_vols = R_vols

    def solve_batch(self, my_net, vnfs_list, R_ids, R_vols, reqs):
        pass

    def solve(self, chain_req, t, sr):
        pass

    def handle_sfc_eviction(self, chain_req, t):
        self.my_net.evict_sfc(chain_req)
        chain_req.used_servers = set()

    def pre_arrival_procedure(self, t):
        pass

    def post_arrival_procedure(self, status, t, chain_req):
        pass

    def reset(self):
        pass


class CloudSolver(Solver):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "CL"

    def cloud_embed(self, chain_req):
        prev = chain_req.entry_point
        cloud_node = "c"
        for pth_id_1 in range(len(self.my_net.paths_links[prev][cloud_node])):
            for pth_id_2 in range(len(self.my_net.paths_links[cloud_node][prev])):
                b1 = self.my_net.get_path_min_bw(prev, cloud_node, pth_id_1, chain_req.T2)
                b2 = self.my_net.get_path_min_bw(cloud_node, prev, pth_id_2, chain_req.T2)
                d1 = self.my_net.get_path_delay(prev, cloud_node, pth_id_1)
                d2 = self.my_net.get_path_delay(cloud_node, prev, pth_id_2)
                if chain_req.vnf_in_rate(0) <= b1 and chain_req.vnf_in_rate(len(chain_req.vnfs)) <= b2:
                    if d1 + d2 <= chain_req.max_delay:
                        chain_bw = 0
                        for ll in self.my_net.paths_links[prev][cloud_node][pth_id_1]:
                            l_obj = self.my_net.g[ll[0]][ll[1]]["li"]
                            l_obj.embed(chain_req, 0)
                            chain_bw = chain_bw + chain_req.vnf_in_rate(0)
                        for ll in self.my_net.paths_links[cloud_node][prev][pth_id_2]:
                            l_obj = self.my_net.g[ll[0]][ll[1]]["li"]
                            l_obj.embed(chain_req, len(chain_req.vnfs))
                            chain_bw = chain_bw + chain_req.vnf_in_rate(len(chain_req.vnfs))
                        return True, chain_bw
        return False, 0

    def solve(self, chain_req, t, sr):
        c_s, c_b = self.cloud_embed(chain_req)
        tr = TestResult()
        if c_s:
            return tr.SU, 0, c_b
        else:
            return tr.SF, 0, 0


class FfSolver(CloudSolver):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "FF"

    def reset(self):
        self.my_net.reset()
        self.my_net.enable_layer_sharing()

    def solve(self, chain_req, t, sr):
        c_s, c_b = self.cloud_embed(chain_req)
        tr = TestResult()
        if c_s:
            return tr.SU, 0, c_b

        downloads = set()
        loc_of = dict()
        cur = chain_req.entry_point
        chain_delay = 0
        chain_bw = 0
        dl_vol = 0
        for i in range(len(chain_req.vnfs)+1):
            st = self.place(chain_req, i, cur, chain_delay)
            if len(st) > 0:
                chain_delay = chain_delay + st[0]
                chain_bw = chain_bw + st[1]
                dl_vol = dl_vol + st[2]
                downloads.update(st[3])
                loc_of[i] = st[4]
                cur = loc_of[i]
            else:
                for ii in loc_of:
                    if loc_of[ii][0] == "e":
                        self.my_net.g.nodes[loc_of[ii]]["nd"].unembed(chain_req, ii)
                self.my_net.evict_sfc(chain_req)
                for ld in downloads:
                    ld.cancel_download()
                return tr.SF, 0, 0

        for ii in loc_of:
            if loc_of[ii][0] == "e":
                self.my_net.g.nodes[loc_of[ii]]["nd"].finalize_layer()
        return tr.SU, dl_vol, chain_bw

    def place(self, chain_req, i, cur, chain_delay):
        if i == len(chain_req.vnfs):
            E = self.my_net.get_all_edge_nodes()
            all_nodes = [cur] + E + ["c"]
        else:
            all_nodes = [chain_req.entry_point]
        for e in all_nodes:
            st = self.place_e(cur, e, chain_req, i, chain_delay)
            if len(st) > 0:
                return st + [e]
        return []

    def place_e(self, loc_i_1, loc_i, req, i, chain_delay):
        cloud_node = "c"
        if i < len(req.vnfs) and loc_i != cloud_node:
            if loc_i[0] == "b":
                return []

            if self.my_net.g.nodes[loc_i]["nd"].cpu_min_avail(req.T2) < req.cpu_req(i):
                return []
            if self.my_net.g.nodes[loc_i]["nd"].ram_min_avail(req.T2) < req.ram_req(i):
                return []
            Rd_ei, d_ei = self.my_net.get_need_storage_layers(loc_i, req, i, req.tau1)
            if self.my_net.g.nodes[loc_i]["nd"].disk_min_avail_no_cache(req.T2) < d_ei:
                return []

        path_id_sel = None
        if loc_i_1 != loc_i:
            for pth_id in range(len(self.my_net.paths_links[loc_i_1][loc_i])):
                b1 = self.my_net.get_path_min_bw(loc_i_1, loc_i, pth_id, req.T2)
                d1 = self.my_net.get_path_delay(loc_i_1, loc_i, pth_id)
                if req.vnf_in_rate(i) <= b1:
                    if chain_delay + d1 <= req.max_delay:
                        path_id_sel = pth_id
                        break
            if path_id_sel is None:
                return []

        downloads = set()
        total_dl_vol = 0
        if i < len(req.vnfs):
            Rd_ei, _ = self.my_net.get_missing_layers(loc_i, req, i, req.tau1)
            for rr in Rd_ei:
                downloaded = False
                for pth_id in range(len(self.my_net.paths_links[loc_i][cloud_node])):
                    b1 = self.my_net.get_path_min_bw(loc_i, cloud_node, pth_id, req.T1)
                    if Rd_ei[rr] / len(req.T1) <= b1:
                        downloaded = True
                        total_dl_vol = total_dl_vol + Rd_ei[rr]
                        layer_download = LayerDownload()
                        downloads.add(layer_download)
                        for tt in req.T1:
                            for ll in self.my_net.paths_links[loc_i][cloud_node][pth_id]:
                                l_obj = self.my_net.g[ll[0]][ll[1]]["li"]
                                layer_download.add_data(tt, l_obj, Rd_ei[rr] / len(req.T1))
                        break
                if not downloaded:
                    for ld in downloads:
                        ld.cancel_download()
                    return []
            self.my_net.g.nodes[loc_i]["nd"].embed(req, i)
            req.used_servers.add(loc_i)

        path_delay = 0
        chain_bw = 0
        if path_id_sel is not None:
            for ll in self.my_net.paths_links[loc_i_1][loc_i][path_id_sel]:
                l_obj = self.my_net.g[ll[0]][ll[1]]["li"]
                l_obj.embed(req, i)
            path_delay = self.my_net.get_path_delay(loc_i_1, loc_i, path_id_sel)
            chain_bw = len(self.my_net.paths_links[loc_i_1][loc_i][path_id_sel]) * req.vnf_in_rate(i)

        return [path_delay, chain_bw, total_dl_vol, downloads]

    def post_arrival_procedure(self, status, t, chain_req):
        for m in self.my_net.g.nodes():
            if m[0] == "e":
                if self.my_net.g.nodes[m]["nd"].disk_avail(t) < 0:
                    print("From {}: delete".format(m))
                    self.my_net.g.nodes[m]["nd"].empty_storage_random(t)


class GreedySolver(FfSolver):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "IGA"

    def place(self, chain_req, i, cur, chain_delay):
        if len(chain_req.vnfs) == i:
            st = self.place_e(cur, chain_req.entry_point, chain_req, i, chain_delay)
            if len(st) > 0:
                return st + ["c"]
            else:
                return []

        if cur == "c":
            st = self.place_e(cur, "c", chain_req, i, chain_delay)
            if len(st) > 0:
                return st + ["c"]

        E = self.my_net.get_all_edge_nodes()
        all_nodes = [cur] + E
        best_node = []
        best_node_metric = 0
        for e in all_nodes:
            T = chain_req.T2
            Rd_ei, d_ei = self.my_net.get_need_storage_layers(e, chain_req, i, chain_req.tau1)
            e_metric_disk = self.my_net.g.nodes[e]["nd"].disk_min_avail_no_cache(T) / d_ei
            e_metric_cpu = self.my_net.g.nodes[e]["nd"].cpu_min_avail(T) / chain_req.cpu_req(i)
            e_metric_ram = self.my_net.g.nodes[e]["nd"].ram_min_avail(T) / chain_req.ram_req(i)
            e_metric_in = np.infty
            if e != cur:
                e_metric_in = self.my_net.get_total_in_bw(e, T) / chain_req.vnf_in_rate(i)
            e_metric_out = self.my_net.get_total_out_bw(e, T) / chain_req.vnf_in_rate(i + 1)
            e_metric = min(
                e_metric_disk, e_metric_cpu, e_metric_ram, e_metric_in, e_metric_out
            )
            if len(best_node) == 0 or e_metric > best_node_metric:
                st = self.place_e(cur, e, chain_req, i, chain_delay)
                if len(st) > 0:
                    best_node = st + [e]
                    best_node_metric = e_metric

        if len(best_node) == 0:
            st = self.place_e(cur, "c", chain_req, i, chain_delay)
            if len(st) > 0:
                return st + ["c"]
            else:
                return []
        else:
            return best_node


class GurobiBatch(Solver):
    def __init__(self):
        super().__init__()
        self.batch = True

    def get_name(self):
        return "B"

    def set_env(self, my_net, R_ids, R_vols):
        self.my_net = my_net
        self.my_net.enable_layer_sharing()
        self.R_ids = R_ids
        self.R_vols = R_vols

    def solve_batch(self, my_net, vnfs_list, R_ids, R_vols, reqs):
        return solve_batch_opt(reqs, self.my_net, self.R_ids, self.R_vols)

    def reset(self):
        self.my_net.reset()
        self.my_net.enable_layer_sharing()


class GurobiSingle(Solver):
    def __init__(self, eviction_strategy="default"):
        super().__init__()
        self.my_net.enable_layer_sharing()
        self.eviction_strategy = eviction_strategy

    def get_name(self):
        return "GrSi"

    def solve(self, chain_req, t, sr):
        return solve_single(self.my_net, self.R_ids, self.R_vols, chain_req)

    def pre_arrival_procedure(self, t):
        for m in self.my_net.g.nodes():
            if m[0] == "e":
                self.my_net.g.nodes[m]["nd"].make_s1()

    def post_arrival_procedure(self, status, t, chain_req):
        for m in self.my_net.g.nodes():
            if m[0] == "e":
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
                    print("From {}: delete".format(m))
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

    def reset(self):
        self.my_net.reset()
        self.my_net.enable_layer_sharing()


class GurobiSingleRelax(Solver):
    def __init__(self, Gamma, bs, eviction_strategy="default", convert_layer=False):
        super().__init__()
        self.eviction_strategy = eviction_strategy
        self.Gamma = Gamma
        self.bw_scaler = bs
        self.convert_layer = convert_layer

    def get_name(self):
        return "RCCO"

    def set_env(self, my_net, R_ids, R_vols):
        self.my_net = my_net
        self.my_net.enable_layer_sharing()
        self.R_ids = R_ids
        self.R_vols = R_vols

    def do_convert_no_share(self, reqs):
        cnt = 0
        R_ids = list()
        R_vols = list()
        for req in reqs:
            for vnf in req.vnfs:
                new_layers = dict()
                for l in vnf.layers:
                    new_layers[cnt] = vnf.layers[l]
                    R_ids.append(cnt)
                    R_vols.append(new_layers[cnt])
                    cnt = cnt + 1
                vnf.layers = new_layers
        return R_ids, R_vols

    def solve(self, chain_req, t, sr):
        rs = RelaxSingle(self.my_net, self.R_ids, self.R_vols, self.Gamma, self.bw_scaler)
        return rs.solve_single_relax(chain_req)

    def post_arrival_procedure(self, status, t, chain_req):
        for m in self.my_net.g.nodes():
            if m[0] == "e":
                if self.eviction_strategy == "popularity_learn":
                    inuse = self.my_net.g.nodes[m]["nd"].get_all_inuse()
                    self.my_net.g.nodes[m]["nd"].p_agent.add_inuse(inuse)
                if self.my_net.g.nodes[m]["nd"].disk_avail(t) < 0:
                    print("From {}: delete".format(m))
                    if self.eviction_strategy == "popularity_learn":
                        self.my_net.g.nodes[m]["nd"].empty_storage_popularity(t)
                    else:
                        self.my_net.g.nodes[m]["nd"].empty_storage_random(t)

    def reset(self):
        self.my_net.reset()
        self.my_net.enable_layer_sharing()