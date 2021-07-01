import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import heapq
from constants import Const
from sfc import LayerDownload


class MyLayer:
    def __init__(self, layer_no, size, chain_user, t, dl_start):
        self.layer_no = layer_no
        self.size = size
        self.chain_users = set()
        self.chain_users.add(chain_user)
        self.avail_from = t
        self.last_used = chain_user.tau2
        self.finalized = False
        self.dl_start = dl_start
        self.unique_used = 1

    def add_user(self, u):
        self.chain_users.add(u)
        self.unique_used = self.unique_used + 1
        if u.tau2 > self.last_used:
            self.last_used = u.tau2

    def remove_user(self, u):
        if u in self.chain_users:
            self.chain_users.remove(u)


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
        return np.random.choice(E, int(sr * len(E)))

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

    def reset(self):
        for n in self.g.nodes():
            self.g.nodes[n]["nd"].reset()
        for e in self.g.edges():
            for j in self.g[e[0]][e[1]]:
                self.g[e[0]][e[1]][j]["li"].reset()


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
        return "{}: {}".format(self.type, self.e1.id)

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

    def layer_avail(self, r, t):
        if self.type[0] == "b":
            return False
        if self.type[0] == "c":
            return True
        if r not in self.layers:
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

    def embed(self, chain_req, i):
        if chain_req not in self.embeds:
            self.embeds[chain_req] = set()
        self.embeds[chain_req].add(i)

    def add_layer(self, R, chain_req):
        for r in R:
            if r in self.layers:
                self.layers[r].add_user(chain_req)
            else:
                self.layers[r] = MyLayer(r, R[r], chain_req, chain_req.tau1, chain_req.arrival_time)

    def add_layer_no_share(self, R, chain_req):
        for r in R:
            self.layers[(r, chain_req)] = MyLayer(r, R[r], chain_req, chain_req.tau1, chain_req.arrival_time)

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
