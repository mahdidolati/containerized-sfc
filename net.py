import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import heapq


class MyNetwork:
    def __init__(self, g):
        self.g = g

    def get_biggest_path(self, c, d):
        h = []
        counter = 0
        heapq.heappush(h, (-np.infty, counter, c, []))
        while True:
            bw, _, n, p = heapq.heappop(h)
            if n == d:
                return -bw, p
            for m in self.g.neighbors(n):
                if m not in p:
                    for j in self.g[n][m]:
                        bw_avail = self.g[n][m][j]["li"].bw_avail()
                        counter = counter + 1
                        pp = list(p)
                        pp.append(n)
                        heapq.heappush(h, (max(bw, -bw_avail), counter, m, pp))


class Link:
    def __init__(self, t, s):
        self.type = t
        self.src = s
        self.bw = 1
        self.embeds = dict()

    def bw_avail(self):
        if self.type == "wired":
            u = 0
            for r in self.embeds:
                for i in self.embeds[r]:
                    u = u + r.vnf_in_rate(i)
            return self.bw - u
        else:
            return self.src.mm_avail()

    def embed(self, chain_req, i):
        if self.type == "wired":
            if chain_req not in self.embeds:
                self.embeds[chain_req] = set()
            self.embeds[chain_req].add(i)
        else:
            self.src.mm_embed(chain_req, i)


class Node:
    def __init__(self, t, loc):
        self.type = t
        self.loc = loc
        self.cpu = 1
        self.ram = 2
        self.disk = 3
        self.mm_bw = 1
        self.layers = dict()
        self.embeds = dict()
        self.mm_embeds = dict()

    def cpu_avail(self):
        u = 0
        for r in self.embeds:
            for i in self.embeds[r]:
                u = u + r.cpu_req(i)
        return self.cpu - u

    def ram_avail(self):
        u = 0
        for r in self.embeds:
            for i in self.embeds[r]:
                u = u + r.ram_req(i)
        return self.ram - u

    def disk_avail(self):
        u = 0
        for r in self.layers:
            u = u + self.layers[r]
        return self.disk - u

    def embed(self, chain_req, i):
        if chain_req not in self.embeds:
            self.embeds[chain_req] = set()
        self.embeds[chain_req].add(i)

    def mm_avail(self):
        u = 0
        for r in self.mm_embeds:
            for i in self.mm_embeds[r]:
                u = u + r.vnf_in_rate(i)
        return self.mm_bw - u

    def mm_embed(self, chain_req, i):
        if chain_req not in self.mm_embeds:
            self.mm_embeds[chain_req] = set()
        self.mm_embeds[chain_req].add(i)


class NetGenerator:
    def __init__(self):
        base_station_loc = [(0, 6), (3, 6), (6, 6), (0, 3), (0, 0), (3, 0), (6, 0)]
        cloud_loc = (10, 3)
        self.g = nx.MultiGraph()
        for n in range(len(base_station_loc)):
            nd = Node("base-station", base_station_loc[n])
            self.g.add_node("b{}".format(n), nd=nd)
        self.edge_num = 15
        for n in range(self.edge_num):
            nd = Node("edge", np.random.uniform(0.5, 5.5, 2))
            self.g.add_node("e{}".format(n), nd=nd)
        nd = Node("cloud", cloud_loc)
        self.g.add_node("c", nd=nd)
        #
        for n in range(len(base_station_loc)):
            c = self.get_closest("b{}".format(n))
            li = Link("wired", self.g.nodes["b{}".format(n)]["nd"])
            self.g.add_edge("b{}".format(n), "e{}".format(c), li=li)
        c = self.get_closest("c")
        li = Link("wired", self.g.nodes["c"]["nd"])
        self.g.add_edge("c", "e{}".format(c), li=li)
        for l in combinations(range(self.edge_num), 2):
            e1 = "e{}".format(l[0])
            e2 = "e{}".format(l[1])
            if np.random.uniform(0, 1.0) < 0.1:
                li = Link("wired", self.g.nodes[e1]["nd"])
                self.g.add_edge(e1, e2, li=li)
            if np.linalg.norm(self.g.nodes[e1]["nd"].loc - self.g.nodes[e2]["nd"].loc) < 2.0:
                li = Link("mmWave", self.g.nodes[e1]["nd"])
                self.g.add_edge(e1, e2, li=li)

    def get_g(self):
        fig, ax = plt.subplots()
        x = []
        y = []
        for n in self.g.nodes():
            x.append(self.g.nodes[n]["nd"].loc[0])
            y.append(self.g.nodes[n]["nd"].loc[1])
        ax.plot(x, y, '.b')
        for n in self.g.nodes():
            ax.annotate(n, self.g.nodes[n]["nd"].loc)
        for e in self.g.edges(data=True):
            for j in self.g[e[0]][e[1]]:
                line_t = 'r-'
                if self.g[e[0]][e[1]][j]["li"].type == "mmWave":
                    line_t = 'b-'
                ax.plot([self.g.nodes[e[0]]["nd"].loc[0], self.g.nodes[e[1]]["nd"].loc[0]],
                        [self.g.nodes[e[0]]["nd"].loc[1], self.g.nodes[e[1]]["nd"].loc[1]], line_t)
        plt.show()
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
