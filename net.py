import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import heapq
from constants import Const
from sfc import LayerDownload


class MyLayer:
    def __init__(self, layer_no, size, chain_user, t):
        self.layer_no = layer_no
        self.size = size
        self.chain_users = set()
        self.chain_users.add(chain_user)
        self.avail_from = t

    def add_user(self, u):
        self.chain_users.add(u)

    def remove_user(self, u):
        self.chain_users.remove(u)


class MyNetwork:
    def __init__(self, g):
        self.g = g

    def get_biggest_path(self, c, d, t, delay_cap=np.infty):
        h = []
        visited = set()
        counter = 0
        heapq.heappush(h, (-np.infty, counter, c, [], 0, []))
        while len(h) > 0:
            bw, cnt, n, cur_path, path_delay, cur_links = heapq.heappop(h)
            if n == d:
                return -bw, path_delay, cur_links
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
        return 0, np.infty, []

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
        R = dict()
        d = 0
        for r in chain_req.vnfs[vnf_i].layers:
            if not self.g.nodes[server]["nd"].layer_avail(r, t):
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
            path_bw, path_delay, links = self.get_biggest_path(server, "c", tt)
            if path_bw < dl_rate:
                layer_download.cancel_download()
                return False, None
            for l in links:
                layer_download.add_data(tt, l, dl_rate)
        return True, layer_download

    def evict_sfc(self, chain_req):
        for n in self.g.nodes():
            self.g.nodes[n]["nd"].evict(chain_req)
        for e in self.g.edges():
            for j in self.g[e[0]][e[1]]:
                self.g[e[0]][e[1]][j]["li"].evict(chain_req)

    def reset(self):
        for n in self.g.nodes():
            self.g.nodes[n]["nd"].reset()
        for e in self.g.edges():
            for j in self.g[e[0]][e[1]]:
                self.g[e[0]][e[1]][j]["li"].reset()


class Link:
    def __init__(self, tp, s, d):
        self.type = tp
        self.src = s
        self.dst = d
        if self.src.id[0] == "c" or self.dst.id[0] == "c":
            self.bw = np.infty
        elif self.type == "wired":
            self.bw = np.random.randint(*Const.LINK_BW)
        else:
            self.bw = 0
        self.delay = 10 * np.linalg.norm(self.src.loc - self.dst.loc)
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
            return self.src.mm_avail(t)

    def embed(self, chain_req, i):
        if self.type == "wired":
            if chain_req not in self.embeds:
                self.embeds[chain_req] = set()
            self.embeds[chain_req].add(i)
        else:
            self.src.mm_embed(chain_req, i)

    def evict(self, chain_req):
        if self.type == "wired":
            if chain_req in self.embeds:
                del self.embeds[chain_req]
        else:
            self.src.mm_evict(chain_req)

    def add_dl(self, t, r):
        if self.type == "wired":
            if t not in self.dl:
                self.dl[t] = 0
            self.dl[t] = self.dl[t] + r
        else:
            self.src.add_mm_dl(t, r)

    def rm_dl(self, t, r):
        if self.type == "wired":
            if t not in self.dl:
                return
            self.dl[t] = self.dl[t] - r
        else:
            self.src.rm_mm_dl(t, r)

    def __str__(self):
        return "{}: {}".format(self.type, self.src.id)

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
        self.mm_bw = np.random.randint(*Const.MM_BW)
        self.layers = dict()
        self.embeds = dict()
        self.mm_embeds = dict()
        self.dl_embeds = dict()

    def reset(self):
        self.layers = dict()
        self.embeds = dict()
        self.mm_embeds = dict()
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
            if t >= self.layers[my_layer].avail_from:
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

    def embed(self, chain_req, i):
        if chain_req not in self.embeds:
            self.embeds[chain_req] = set()
        self.embeds[chain_req].add(i)

    def add_layer(self, R, chain_req):
        for r in R:
            if r in self.layers:
                self.layers[r].add_user(chain_req)
            else:
                self.layers[r] = MyLayer(r, R[r], chain_req, chain_req.tau1)

    def evict(self, chain_req):
        if chain_req in self.embeds:
            del self.embeds[chain_req]

    def mm_avail(self, t):
        u = 0
        for r in self.mm_embeds:
            if r.tau1 <= t <= r.tau2:
                for i in self.mm_embeds[r]:
                    u = u + r.vnf_in_rate(i)
        if t in self.dl_embeds:
            u = u + self.dl_embeds[t]
        return self.mm_bw - u

    def mm_embed(self, chain_req, i):
        if chain_req not in self.mm_embeds:
            self.mm_embeds[chain_req] = set()
        self.mm_embeds[chain_req].add(i)

    def mm_evict(self, chain_req):
        if chain_req in self.mm_embeds:
            del self.mm_embeds[chain_req]

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
        self.g = nx.MultiGraph()
        for n in range(len(base_station_loc)):
            n_id = "b{}".format(n)
            nd = Node("base-station", base_station_loc[n], n_id)
            self.g.add_node(n_id, nd=nd)
        self.edge_num = 15
        for n in range(self.edge_num):
            n_id = "e{}".format(n)
            nd = Node("edge", np.random.uniform(0.5, 5.5, 2), n_id)
            self.g.add_node(n_id, nd=nd)
        n_id = "c"
        nd = Node("cloud", cloud_loc, n_id)
        self.g.add_node(n_id, nd=nd)
        #
        for n in range(len(base_station_loc)):
            c = self.get_closest("b{}".format(n))
            li = Link("wired", self.g.nodes["b{}".format(n)]["nd"], self.g.nodes["e{}".format(c)]["nd"])
            self.g.add_edge("b{}".format(n), "e{}".format(c), li=li)
        c = self.get_closest("c")
        li = Link("wired", self.g.nodes["c"]["nd"], self.g.nodes["e{}".format(c)]["nd"])
        self.g.add_edge("c", "e{}".format(c), li=li)
        for l in combinations(range(self.edge_num), 2):
            e1 = "e{}".format(l[0])
            e2 = "e{}".format(l[1])
            if np.random.uniform(0, 1.0) < 0.1:
                li = Link("wired", self.g.nodes[e1]["nd"], self.g.nodes[e2]["nd"])
                self.g.add_edge(e1, e2, li=li)
            if np.linalg.norm(self.g.nodes[e1]["nd"].loc - self.g.nodes[e2]["nd"].loc) < 2.0:
                li = Link("mmWave", self.g.nodes[e1]["nd"], self.g.nodes[e2]["nd"])
                self.g.add_edge(e1, e2, li=li)

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
        for n in range(self.edge_num):
            l2 = self.g.nodes["e{}".format(n)]["nd"].loc
            d = np.linalg.norm(l1 - l2)
            if closest is None or d < ds:
                ds = d
                closest = n
        return closest
