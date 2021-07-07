import heapq


class PopularityEntity:
    def __init__(self, layer_id, layer_val):
        self.layer_id = layer_id
        self.layer_size = layer_val
        self.layer_val = layer_val

    def update_val(self, layer_val):
        self.layer_val = self.layer_val + layer_val


class Solver:
    def __init__(self, my_net):
        self.my_net = my_net

    def sort_nodes_disk(self, all_nodes, chain_req, i):
        h = []
        counter = 0
        for n in all_nodes:
            R, d = self.my_net.get_missing_layers(n, chain_req, i, chain_req.tau1)
            heapq.heappush(h, (d, counter, n))
            counter = counter + 1
        return h

    def usable_node(self, s, c, chain_req, i, t, delay_budget):
        for tt in range(chain_req.tau1, chain_req.tau2 + 1):
            if self.my_net.g.nodes[c]["nd"].cpu_avail(tt) < chain_req.cpu_req(i) or \
                    self.my_net.g.nodes[c]["nd"].ram_avail(tt) < chain_req.ram_req(i):
                return False, 0
        if c[0] == "b":
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
        self.update_layer_popularity(chain_req)
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
            chain_req.used_servers.add(m)
            R, d = self.my_net.get_missing_layers(m, chain_req, i, chain_req.tau1)
            max_del = 0
            for tt in range(chain_req.tau1, chain_req.tau2 + 1):
                max_del = max(max_del, abs(self.my_net.g.nodes[m]["nd"].disk_avail(tt) - d))
            if max_del > 0:
                if m not in mark_del_layer:
                    mark_del_layer[m] = set()
                mark_del_layer[m].update(self.my_net.g.nodes[m]["nd"].get_unused(max_del))
            layer_download_vol = layer_download_vol + d
            if len(R) > 0:
                dl_result, dl_obj = self.my_net.do_layer_dl_test(m, R, d, t, chain_req.tau1 - 1)
                active_dls.append(dl_obj)
                if self.my_net.share_layer:
                    self.my_net.g.nodes[m]["nd"].add_layer(R, chain_req)
                else:
                    self.my_net.g.nodes[m]["nd"].add_layer_no_share(R, chain_req)
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
                if self.delete_layer(self.my_net.g.nodes[m]["nd"].layers[l], t):
                    to_be_delete.add((m, l))
        for m, l in to_be_delete:
            del self.my_net.g.nodes[m]["nd"].layers[l]

    def delete_layer(self, target_layer, t):
        return True

    def pre_fetch_layers(self, t):
        return 0.0

    def update_layer_popularity(self, chain_req):
        pass


class NoShareSolver(Solver):
    def __init__(self, my_net, layer_del_th):
        super().__init__(my_net)
        self.layer_del_th = layer_del_th
        self.my_net.share_layer = False

    def get_name(self):
        return "NS-{}".format(self.layer_del_th)

    def reset(self):
        self.my_net.reset()
        self.my_net.share_layer = False

    def delete_layer(self, target_layer, t):
        if len(target_layer.chain_users) == 0:
            if t - target_layer.last_used > self.layer_del_th:
                return True
        return False


class ShareSolver(Solver):
    def __init__(self, my_net, layer_del_th):
        super().__init__(my_net)
        self.layer_del_th = layer_del_th
        self.my_net.share_layer = True

    def get_name(self):
        return "S-{}".format(self.layer_del_th)

    def reset(self):
        self.my_net.reset()
        self.my_net.share_layer = True

    def delete_layer(self, target_layer, t):
        if len(target_layer.chain_users) == 0:
            if t - target_layer.last_used > self.layer_del_th:
                return True
        return False


class PopularitySolver(Solver):
    def __init__(self, my_net, popularity_th):
        super().__init__(my_net)
        self.popularity_th = popularity_th
        self.my_net.share_layer = True

    def get_name(self):
        return "P-{}".format(self.popularity_th)

    def reset(self):
        self.my_net.reset()
        self.my_net.share_layer = True

    def delete_layer(self, target_layer, t):
        if len(target_layer.chain_users) == 0:
            if target_layer.unique_used < self.popularity_th:
                return True
            else:
                target_layer.unique_used = target_layer.unique_used - 1
        return False


class ProactiveSolver(Solver):
    def __init__(self, my_net, disk_thresh, unavail_ahead):
        super().__init__(my_net)
        self.my_net.share_layer = True
        self.popularity_rec = list()
        self.popularity_obj = dict()
        self.disk_thresh = disk_thresh
        self.unavail_ahead = unavail_ahead

    def get_name(self):
        return "A-{},{}".format(self.disk_thresh, self.unavail_ahead)

    def reset(self):
        self.my_net.reset()
        self.my_net.share_layer = True

    def delete_layer(self, target_layer, t):
        if len(target_layer.chain_users) == 0:
            if target_layer.last_used + self.unavail_ahead < t:
                return True
        return False

    def pre_fetch_layers(self, t):
        N1 = self.my_net.get_random_edge_nodes(1.0)
        dl_vol = 0
        for n in N1:
            for r in range(len(self.popularity_rec)):
                if self.disk_thresh <= self.my_net.g.nodes[n]["nd"].disk_avail_ratio(t):
                    if self.my_net.g.nodes[n]["nd"].layer_avail(self.popularity_rec[r].layer_id, t+self.unavail_ahead):
                        R = [self.popularity_rec[r].layer_id]
                        d = self.popularity_rec[r].layer_size
                        dl_result, dl_obj = self.my_net.do_layer_dl_test(n, R, d, t, t+self.unavail_ahead)
                        if not dl_result:
                            continue
                        dl_vol = dl_vol + d
                        self.my_net.g.nodes[n]["nd"].add_proactive_layer(
                                                        self.popularity_rec[r].layer_id,
                                                        self.popularity_rec[r].layer_size,
                                                        t, t+self.unavail_ahead+1
                                                    )
        return dl_vol

    def update_layer_popularity(self, chain_req):
        for i in range(len(chain_req.vnfs)):
            for r in chain_req.vnfs[i].layers:
                if r not in self.popularity_obj:
                    self.popularity_obj[r] = PopularityEntity(r, chain_req.vnfs[i].layers[r])
                    self.popularity_rec.append(self.popularity_obj[r])
                else:
                    self.popularity_obj[r].update_val(chain_req.vnfs[i].layers[r])
        self.popularity_rec.sort(key=lambda x: x.layer_val, reverse=True)


class StorageAwareSolver(Solver):
    def __init__(self, my_net):
        super().__init__(my_net)
        self.my_net.share_layer = True

    def get_name(self):
        return "SA"

    def usable_node(self, s, c, chain_req, i, t, delay_budget):
        for tt in range(chain_req.tau1, chain_req.tau2 + 1):
            if self.my_net.g.nodes[c]["nd"].cpu_avail(tt) < chain_req.cpu_req(i) or \
                    self.my_net.g.nodes[c]["nd"].ram_avail(tt) < chain_req.ram_req(i):
                return False, 0
        if c[0] == "b":
            return False, 0
        R, d = self.my_net.get_missing_layers(c, chain_req, i, chain_req.tau1)
        for tt in range(chain_req.tau1, chain_req.tau2 + 1):
            if self.my_net.g.nodes[c]["nd"].disk_avail_no_cache(tt) < d:
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

    def reset(self):
        self.my_net.reset()
        self.my_net.share_layer = True

    def delete_layer(self, target_layer, t):
        return False
