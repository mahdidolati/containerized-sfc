import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import heapq
from constants import Const
from sfc import LayerDownload


class MyLayer:
    def __init__(self, layer_no, size, dl_start, avail_from):
        self.layer_no = layer_no
        self.size = size
        self.dl_start = dl_start
        self.avail_from = avail_from
        self.last_used = avail_from
        self.unique_used = 0
        self.chain_users = set()
        self.finalized = False
        # for storage aware-ness
        self.marked_needed = False
        self.marked_delete = False

    def add_user(self, u):
        self.marked_needed = True
        self.chain_users.add(u)
        self.unique_used = self.unique_used + 1
        if u.tau2 > self.last_used:
            self.last_used = u.tau2

    def remove_user(self, u):
        if u in self.chain_users:
            self.chain_users.remove(u)
        self.marked_needed = len(self.chain_users) > 0


class MyNetwork:
    def __init__(self, g, share_layer=True):
        self.g = g
        self.share_layer = share_layer

    def get_biggest_path(self, c, d, t, delay_cap=np.infty):
        h = []
        visited = set()
        counter = 0
        heapq.heappush(h, (-np.infty, counter, c, [], 0, []))
        while len(h) > 0:
            bw, cnt, n, cur_path, path_delay, cur_links = heapq.heappop(h)
            if n == d:
                return -bw, path_delay, cur_path, cur_links
            if n in visited:
                continue
            visited.add(n)
            for m in self.g.neighbors(n):
                if m not in visited and m not in cur_path and self.g.nodes[m]["nd"].id[0] != "b":
                    for j in self.g[n][m]:
                        bw_avail = self.g[n][m][j]["li"].bw_avail(t)
                        link_delay = self.g[n][m][j]["li"].delay
                        if path_delay + link_delay <= delay_cap:
                            counter = counter + 1
                            new_path = list(cur_path)
                            new_path.append(n)
                            new_links = list(cur_links)
                            new_links.append(self.g[n][m][j]["li"])
                            heapq.heappush(h, (max(bw, -bw_avail),
                                               counter, m, new_path,
                                               path_delay + link_delay,
                                               new_links))
        return 0, np.infty, [], []

    def get_a_path(self, c, d, t, excluded=None):
        if excluded is None:
            excluded = []
        h = []
        visited = set()
        counter = 0
        heapq.heappush(h, (-np.infty, counter, c, [], []))
        while len(h) > 0:
            bw, cnt, n, cur_path, cur_links = heapq.heappop(h)
            if n == d:
                return -bw, cur_path, cur_links
            if n in visited:
                continue
            visited.add(n)
            for m in self.g.neighbors(n):
                if m not in visited and m not in cur_path and self.g.nodes[m]["nd"].id[0] != "b":
                    for j in self.g[n][m]:
                        cur_link = self.g[n][m][j]["li"]
                        if (n, m, j) not in excluded:
                            bw_avail = cur_link.bw_avail(t)
                            counter = counter + 1
                            new_path = list(cur_path)
                            new_path.append(n)
                            new_links = list(cur_links)
                            new_links.append((n, m, j))
                            heapq.heappush(h, (max(bw, -bw_avail),
                                               counter, m, new_path, new_links))
        return 0, [], []

    def pre_compute_paths(self, n, t):
        cur_path = []
        bb, pp, ll = self.get_a_path(n, "c", t)
        cur_path.append(ll)
        for l in ll:
            bb2, pp2, ll2 = self.get_a_path(n, "c", t, [l])
            if bb2 > 0:
                cur_path.append(ll2)
                break
        return cur_path

    def get_closest(self, n):
        bestDelay = np.infty
        bestNeighbor = None
        for m in self.g.neighbors(n):
            for j in self.g[n][m]:
                if bestNeighbor is None or bestDelay > self.g[m][n][j]["delay"]:
                    bestDelay = self.g[m][n][j]["li"].delay
                    bestNeighbor = (n, m, j)
        return bestNeighbor

    def get_random_base_state(self):
        C = list()
        for n in self.g.nodes():
            if n[0] == "b":
                C.append(n)
        return np.random.choice(C)

    def get_random_edge_nodes(self, sr):
        E = list()
        for n in self.g.nodes():
            if n[0] == "e":
                E.append(n)
        return np.random.choice(E, int(sr * len(E)), replace=False)

    def get_all_edge_nodes(self):
        E = list()
        for n in self.g.nodes():
            if n[0] == "e":
                E.append(n)
        return E

    def get_missing_layers(self, server, chain_req, vnf_i, t):
        if self.share_layer:
            return self.get_missing_layers_w_share(server, chain_req, vnf_i, t)
        else:
            return self.get_missing_layers_no_share(server, chain_req, vnf_i, t)

    def get_missing_layers_w_share(self, server, chain_req, vnf_i, t):
        R = dict()
        d = 0
        for r in chain_req.vnfs[vnf_i].layers:
            if not self.g.nodes[server]["nd"].layer_avail(r, t):
                R[r] = chain_req.vnfs[vnf_i].layers[r]
                d = d + R[r]
        return R, d

    def get_missing_layers_no_share(self, server, chain_req, vnf_i, t):
        R = dict()
        d = 0
        for r in chain_req.vnfs[vnf_i].layers:
            if not self.g.nodes[server]["nd"].layer_avail_no_share(r, chain_req, t):
                R[r] = chain_req.vnfs[vnf_i].layers[r]
                d = d + R[r]
        return R, d

    def do_layer_dl_test(self, server, candid_layers, volume, start_t, end_t):
        # If an ongoing download exists but can not prepare the layer before end_t+1, return False
        for c_layer in candid_layers:
            if c_layer in self.g.nodes[server]["nd"].layers:
                if self.g.nodes[server]["nd"].layers[c_layer].avail_from > end_t + 1:
                    return False, None
        # Try to download all layers
        dl_rate = volume / (end_t - start_t + 1)
        layer_download = LayerDownload()
        for tt in range(start_t, end_t + 1):
            path_bw, path_delay, path_nodes, links = self.get_biggest_path(server, "c", tt)
            if path_bw < dl_rate:
                layer_download.cancel_download()
                return False, None
            for l in links:
                layer_download.add_data(tt, l, dl_rate)
        return True, layer_download

    def get_min_bw(self, links, t1, t2):
        min_bw = np.infty
        for t in range(t1, t2+1):
            for l in links:
                if l.bw_avail(t) < min_bw:
                    min_bw = l.bw_avail(t)
        return min_bw

    def evict_sfc(self, chain_req):
        for n in self.g.nodes():
            self.g.nodes[n]["nd"].evict(chain_req)
        for e in self.g.edges():
            for j in self.g[e[0]][e[1]]:
                self.g[e[0]][e[1]][j]["li"].evict(chain_req)
        to_be_delete = set()
        for m in chain_req.used_servers:
            for l in self.g.nodes[m]["nd"].layers:
                if not self.g.nodes[m]["nd"].layers[l].finalized:
                    to_be_delete.add((m, l))
        for m, l in to_be_delete:
            del self.g.nodes[m]["nd"].layers[l]

    def get_link_sets(self):
        Lw = list()
        Lm = list()
        L_iii = dict()
        for e in self.g.edges():
            for j in self.g[e[0]][e[1]]:
                L_iii[self.g[e[0]][e[1]][j]["li"]] = (e[0], e[1], j)
                if self.g[e[0]][e[1]][j]["li"].type == "wired":
                    Lw.append((e[0], e[1], j))
                else:
                    Lm.append((e[0], e[1], j))
        return Lw, Lm, L_iii

    def get_all_base_stations(self):
        B = list()
        for n in self.g.nodes():
            if n[0] == "b":
                B.append(n)
        return B

    def reset(self):
        for n in self.g.nodes():
            self.g.nodes[n]["nd"].reset()
        for e in self.g.edges():
            for j in self.g[e[0]][e[1]]:
                self.g[e[0]][e[1]][j]["li"].reset()

    def print(self):
        for n in self.g.nodes():
            ss = self.g.nodes[n]["nd"].cpu
            rr = self.g.nodes[n]["nd"].ram
            dd = self.g.nodes[n]["nd"].disk
            mm = self.g.nodes[n]["nd"].mm_bw_tx
            print("Node: {}, cpu: {}, ram: {}, disk: {}, mm: {}".format(n, ss, rr, dd, mm))
        for e in self.g.edges():
            for j in self.g[e[0]][e[1]]:
                tt = self.g[e[0]][e[1]][j]["li"].type
                bb = self.g[e[0]][e[1]][j]["li"].bw
                dd = self.g[e[0]][e[1]][j]["li"].delay
                print("Link({}): {} -- {} -- {}, bw: {}, dl: {}".format(tt, e[0], e[1], j, bb, dd))


class Link:
    def __init__(self, tp, s, d):
        self.type = tp
        self.e1 = s
        self.e2 = d
        if self.e1.id[0] == "c" or self.e2.id[0] == "c":
            self.bw = np.infty
            self.delay = 100 * np.linalg.norm(self.e1.loc - self.e2.loc)
        elif self.type == "wired":
            self.delay = 10 * np.linalg.norm(self.e1.loc - self.e2.loc)
            self.bw = np.random.randint(*Const.LINK_BW)
        else:
            self.delay = 10 * np.linalg.norm(self.e1.loc - self.e2.loc)
            self.bw = 0
        # print(self.delay)
        self.embeds = dict()
        self.dl = dict()

    def reset(self):
        self.embeds = dict()
        self.dl = dict()

    def bw_avail(self, t):
        if self.type == "wired":
            u = 0
            for r in self.embeds:
                if r.tau1 <= t <= r.tau2:
                    for i in self.embeds[r]:
                        u = u + r.vnf_in_rate(i)
            if t in self.dl:
                u = u + self.dl[t]
            return self.bw - u
        else:
            return min(self.e1.mm_tx_avail(t), self.e2.mm_rx_avail(t))

    def embed(self, chain_req, i):
        if self.type == "wired":
            if chain_req not in self.embeds:
                self.embeds[chain_req] = set()
            self.embeds[chain_req].add(i)
        else:
            self.e1.mm_embed_tx(chain_req, i)
            self.e2.mm_embed_rx(chain_req, i)

    def evict(self, chain_req):
        if self.type == "wired":
            if chain_req in self.embeds:
                del self.embeds[chain_req]
        else:
            self.e1.mm_evict(chain_req)
            self.e2.mm_evict(chain_req)

    def add_dl(self, t, r):
        if self.type == "wired":
            if t not in self.dl:
                self.dl[t] = 0
            self.dl[t] = self.dl[t] + r
        else:
            self.e1.add_mm_dl(t, r)

    def rm_dl(self, t, r):
        if self.type == "wired":
            if t not in self.dl:
                return
            self.dl[t] = self.dl[t] - r
        else:
            self.e1.rm_mm_dl(t, r)

    def __str__(self):
        return "{},{},{}".format(self.type, self.e1.id, self.e2.id)

    def __repr__(self):
        return self.__str__()


class Node:
    def __init__(self, t, loc, id):
        self.id = id
        self.type = t
        self.loc = loc
        if self.type[0] == "b":
            self.cpu = 0
            self.ram = 0
            self.disk = 0
        elif self.type[0] == "e":
            self.cpu = np.random.randint(*Const.SERVER_CPU)
            self.ram = np.random.randint(*Const.SERVER_RAM)
            self.disk = np.random.randint(*Const.SERVER_DISK)
        else:
            self.cpu = np.infty
            self.ram = np.infty
            self.disk = np.infty
        self.mm_bw_tx = np.random.randint(*Const.MM_BW)
        self.mm_bw_rx = np.random.randint(*Const.MM_BW)
        self.layers = dict()
        self.embeds = dict()
        self.mm_embeds_tx = dict()
        self.mm_embeds_rx = dict()
        self.dl_embeds = dict()

    def reset(self):
        self.layers = dict()
        self.embeds = dict()
        self.mm_embeds_tx = dict()
        self.mm_embeds_rx = dict()
        self.dl_embeds = dict()

    def cpu_avail(self, t):
        if self.type[0] == "b":
            return 0
        if self.type[0] == "c":
            return np.infty
        u = 0
        for r in self.embeds:
            if r.tau1 <= t <= r.tau2:
                for i in self.embeds[r]:
                    u = u + r.cpu_req(i)
        return self.cpu - u

    def ram_avail(self, t):
        if self.type[0] == "b":
            return 0
        if self.type[0] == "c":
            return np.infty
        u = 0
        for r in self.embeds:
            if r.tau1 <= t <= r.tau2:
                for i in self.embeds[r]:
                    u = u + r.ram_req(i)
        return self.ram - u

    def disk_avail(self, t):
        if self.type[0] == "b":
            return 0
        if self.type[0] == "c":
            return np.infty
        u = 0
        for my_layer in self.layers:
            if t >= self.layers[my_layer].dl_start:
                u = u + self.layers[my_layer].size
        return self.disk - u

    def disk_avail_no_cache(self, t):
        if self.type[0] == "b":
            return 0
        if self.type[0] == "c":
            return np.infty
        u = 0
        for my_layer in self.layers:
            if len(self.layers[my_layer].chain_users) > 0 or self.layers[my_layer].marked_needed:
                if t >= self.layers[my_layer].dl_start:
                    u = u + self.layers[my_layer].size
        return self.disk - u

    def has_unused_layer(self, t):
        if self.type[0] == "b":
            return False
        if self.type[0] == "c":
            return False
        for my_layer in self.layers:
            if len(self.layers[my_layer].chain_users) == 0 or not self.layers[my_layer].marked_needed:
                return True
        return False

    def get_all_unused(self):
        vol = 0
        unused = set()
        for my_layer in self.layers:
            if len(self.layers[my_layer].chain_users) == 0 and \
                    not self.layers[my_layer].marked_needed and \
                    not self.layers[my_layer].marked_delete:
                unused.add(my_layer)
                vol = vol + self.layers[my_layer].size
        return vol, unused


    def get_unused_for_del(self, max_del):
        unused = set()
        for my_layer in self.layers:
            if len(self.layers[my_layer].chain_users) == 0 and \
                    not self.layers[my_layer].marked_needed and \
                    not self.layers[my_layer].marked_delete:
                unused.add(my_layer)
                max_del = max_del - self.layers[my_layer].size
                self.layers[my_layer].marked_delete = True
            if max_del <= 0:
                break
        return unused

    def disk_avail_ratio(self, t):
        a = self.disk_avail(t)
        return a / self.disk

    def layer_avail(self, r, t):
        if self.type[0] == "b":
            return False
        if self.type[0] == "c":
            return True
        if r not in self.layers:
            return False
        if self.layers[r].marked_delete:
            if not self.swap_layer(r):
                return False
        return t >= self.layers[r].avail_from

    def layer_avail_no_share(self, r, chain_req, t):
        if self.type[0] == "b":
            return False
        if self.type[0] == "c":
            return True
        if (r, chain_req) not in self.layers:
            return False
        return t >= self.layers[(r, chain_req)].avail_from

    def swap_layer(self, r):
        del_vol = 0
        to_del = set()
        for rr in self.layers:
            if len(self.layers[rr].chain_users) == 0 and\
                    not self.layers[rr].marked_needed and\
                    not self.layers[rr].marked_delete:
                del_vol = del_vol + self.layers[rr].size
                to_del.add(rr)
            if del_vol >= self.layers[r].size:
                break
        if del_vol < self.layers[r].size:
            return False
        self.layers[r].marked_delete = False
        for rr in to_del:
            self.layers[rr].marked_delete = True
        return True

    def embed(self, chain_req, i):
        if chain_req not in self.embeds:
            self.embeds[chain_req] = set()
        self.embeds[chain_req].add(i)

    def add_proactive_layer(self, layer_id, layer_size, dl_start, avail_from):
        if layer_id not in self.layers:
            ml = MyLayer(layer_id, layer_size, dl_start, avail_from)
            ml.finalized = True
            self.layers[layer_id] = ml

    # do not delete if while admitting a req
    def mark_needed(self, chain_req, i):
        for r in chain_req.vnfs[i].layers:
            if r in self.layers:
                self.layers[r].marked_needed = True

    def mark_no_need(self, chain_req):
        for i in range(len(chain_req.vnfs)):
            for r in chain_req.vnfs[i].layers:
                if r in self.layers:
                    self.layers[r].marked_needed = False

    def add_layer(self, R, chain_req, mark_fin=False):
        for r in R:
            if r in self.layers:
                self.layers[r].add_user(chain_req)
            else:
                self.layers[r] = MyLayer(r, R[r], chain_req.arrival_time, chain_req.tau1)
                self.layers[r].add_user(chain_req)
            if mark_fin:
                self.layers[r].finalized = True

    def add_layer_no_share(self, R, chain_req):
        for r in R:
            self.layers[(r, chain_req)] = MyLayer(r, R[r], chain_req.arrival_time, chain_req.tau1)
            self.layers[(r, chain_req)].add_user(chain_req)

    def evict(self, chain_req):
        if chain_req in self.embeds:
            del self.embeds[chain_req]

    def mm_tx_avail(self, t):
        u = 0
        for r in self.mm_embeds_tx:
            if r.tau1 <= t <= r.tau2:
                for i in self.mm_embeds_tx[r]:
                    # Transmitted traffic in the current node towards the location of i-th VNF of chain r
                    u = u + r.vnf_in_rate(i)
        if t in self.dl_embeds:
            u = u + self.dl_embeds[t]
        return self.mm_bw_tx - u

    def mm_rx_avail(self, t):
        u = 0
        for r in self.mm_embeds_rx:
            if r.tau1 <= t <= r.tau2:
                for i in self.mm_embeds_rx[r]:
                    # Transmitted traffic in the current node towards the location of i-th VNF of chain r
                    u = u + r.vnf_in_rate(i)
        if t in self.dl_embeds:
            u = u + self.dl_embeds[t]
        return self.mm_bw_rx - u

    def mm_embed_tx(self, chain_req, i):
        if chain_req not in self.mm_embeds_tx:
            self.mm_embeds_tx[chain_req] = set()
        self.mm_embeds_tx[chain_req].add(i)

    def mm_embed_rx(self, chain_req, i):
        if chain_req not in self.mm_embeds_rx:
            self.mm_embeds_rx[chain_req] = set()
        self.mm_embeds_rx[chain_req].add(i)

    def mm_evict(self, chain_req):
        if chain_req in self.mm_embeds_tx:
            del self.mm_embeds_tx[chain_req]
        if chain_req in self.mm_embeds_rx:
            del self.mm_embeds_rx[chain_req]

    def add_mm_dl(self, t, r):
        if t not in self.dl_embeds:
            self.dl_embeds[t] = 0
        self.dl_embeds[t] = self.dl_embeds[t] + r

    def rm_mm_dl(self, t, r):
        if t not in self.dl_embeds:
            return
        self.dl_embeds[t] = self.dl_embeds[t] - r


class NetGenerator:
    def __init__(self):
        base_station_loc = [(0, 6), (3, 6), (6, 6), (0, 3), (0, 0), (3, 0), (6, 0)]
        cloud_loc = (10, 3)
        self.g = nx.MultiDiGraph()
        for n in range(len(base_station_loc)):
            n_id = "b{}".format(n)
            nd = Node("base-station", base_station_loc[n], n_id)
            self.g.add_node(n_id, nd=nd)
        self.e_node_num = 15
        for n in range(self.e_node_num):
            n_id = "e{}".format(n)
            nd = Node("edge", np.random.uniform(0.5, 5.5, 2), n_id)
            self.g.add_node(n_id, nd=nd)
        n_id = "c"
        nd = Node("cloud", cloud_loc, n_id)
        self.g.add_node(n_id, nd=nd)
        #
        for n in range(len(base_station_loc)):
            e1 = "b{}".format(n)
            e2 = "e{}".format(self.get_closest("b{}".format(n)))
            li1 = Link("wired", self.g.nodes[e1]["nd"], self.g.nodes[e2]["nd"])
            li2 = Link("wired", self.g.nodes[e2]["nd"], self.g.nodes[e1]["nd"])
            self.g.add_edge(e1, e2, li=li1)
            self.g.add_edge(e2, e1, li=li2)

        e1 = "c"
        e2 = "e{}".format(self.get_closest(e1))
        li1 = Link("wired", self.g.nodes[e1]["nd"], self.g.nodes[e2]["nd"])
        li2 = Link("wired", self.g.nodes[e2]["nd"], self.g.nodes[e1]["nd"])
        self.g.add_edge(e1, e2, li=li1)
        self.g.add_edge(e2, e1, li=li2)
        #
        for l in combinations(range(self.e_node_num), 2):
            e1 = "e{}".format(l[0])
            e2 = "e{}".format(l[1])
            if np.random.uniform(0, 1.0) < Const.WIRE_LINK_PR:
                li1 = Link("wired", self.g.nodes[e1]["nd"], self.g.nodes[e2]["nd"])
                li2 = Link("wired", self.g.nodes[e2]["nd"], self.g.nodes[e1]["nd"])
                self.g.add_edge(e1, e2, li=li1)
                self.g.add_edge(e2, e1, li=li2)
            if np.linalg.norm(self.g.nodes[e1]["nd"].loc - self.g.nodes[e2]["nd"].loc) < Const.MM_MAX_DIST:
                li1 = Link("mmWave", self.g.nodes[e1]["nd"], self.g.nodes[e2]["nd"])
                li2 = Link("mmWave", self.g.nodes[e2]["nd"], self.g.nodes[e1]["nd"])
                self.g.add_edge(e1, e2, li=li1)
                self.g.add_edge(e2, e1, li=li2)

    def get_g(self):
        # fig, ax = plt.subplots()
        # x = []
        # y = []
        # for n in self.g.nodes():
        #     x.append(self.g.nodes[n]["nd"].loc[0])
        #     y.append(self.g.nodes[n]["nd"].loc[1])
        # ax.plot(x, y, '.b')
        # for n in self.g.nodes():
        #     ax.annotate(n, self.g.nodes[n]["nd"].loc)
        # for e in self.g.edges(data=True):
        #     for j in self.g[e[0]][e[1]]:
        #         line_t = 'r-'
        #         if self.g[e[0]][e[1]][j]["li"].type == "mmWave":
        #             line_t = 'b-'
        #         ax.plot([self.g.nodes[e[0]]["nd"].loc[0], self.g.nodes[e[1]]["nd"].loc[0]],
        #                 [self.g.nodes[e[0]]["nd"].loc[1], self.g.nodes[e[1]]["nd"].loc[1]], line_t)
        # plt.show()
        return MyNetwork(self.g)

    def get_closest(self, b):
        ds = np.infty
        closest = None
        l1 = self.g.nodes[b]["nd"].loc
        for n in range(self.e_node_num):
            l2 = self.g.nodes["e{}".format(n)]["nd"].loc
            d = np.linalg.norm(l1 - l2)
            if closest is None or d < ds:
                ds = d
                closest = n
        return closest
