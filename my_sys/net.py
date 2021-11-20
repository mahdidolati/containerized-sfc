import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import heapq
from constants import Const
from sfc import LayerDownload
from q_learn import QLearn
from random import shuffle


class MyLayer:
    def __init__(self, layer_no, size, dl_start, avail_from):
        self.layer_no = layer_no
        self.size = size
        self.dl_start = dl_start
        self.avail_from = avail_from
        self.last_used = avail_from
        self.chain_users = set()

    def add_user(self, u):
        self.chain_users.add(u)
        if u.tau2 > self.last_used:
            self.last_used = u.tau2

    def remove_user(self, u):
        if u in self.chain_users:
            self.chain_users.remove(u)


class MyNetwork:
    def __init__(self, g, share_layer=True):
        self.g = g
        if share_layer:
            self.enable_layer_sharing()
        else:
            self.disable_layer_sharing()

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
                if m not in visited and m not in cur_path:
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

    def enable_layer_sharing(self):
        for n in self.g.nodes():
            if n[0] == "e":
                self.g.nodes[n]["nd"].sharing = True

    def disable_layer_sharing(self):
        for n in self.g.nodes():
            if n[0] == "e":
                self.g.nodes[n]["nd"].sharing = False

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
        R = dict()
        d = 0
        for r in chain_req.vnfs[vnf_i].layers:
            if not self.g.nodes[server]["nd"].layer_avail(r, t, chain_req):
                R[r] = chain_req.vnfs[vnf_i].layers[r]
                d = d + R[r]
        return R, d

    def get_need_storage_layers(self, server, chain_req, vnf_i, t):
        R = dict()
        for r in chain_req.vnfs[vnf_i].layers:
            if not self.g.nodes[server]["nd"].layer_avail(r, t) \
                    or not self.g.nodes[server]["nd"].layer_inuse(r):
                R[r] = chain_req.vnfs[vnf_i].layers[r]
        return R

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

    def get_link_sets(self):
        Lw = list()
        L_iii = dict()
        for e in self.g.edges():
            for j in self.g[e[0]][e[1]]:
                L_iii[self.g[e[0]][e[1]][j]["li"]] = (e[0], e[1], j)
                Lw.append((e[0], e[1], j))
        return Lw, L_iii

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
            print("Node: {}, cpu: {}, ram: {}, disk: {}".format(n, ss, rr, dd))
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
        else:
            self.delay = 10 * np.linalg.norm(self.e1.loc - self.e2.loc)
            self.bw = np.random.randint(*Const.LINK_BW)
        # print(self.delay)
        self.embeds = dict()
        self.dl = dict()

    def reset(self):
        self.embeds = dict()
        self.dl = dict()

    def bw_avail(self, t):
        u = 0
        for r in self.embeds:
            if r.tau1 <= t <= r.tau2:
                for i in self.embeds[r]:
                    u = u + r.vnf_in_rate(i)
        if t in self.dl:
            u = u + self.dl[t]
        return self.bw - u

    def embed(self, chain_req, i):
        if chain_req not in self.embeds:
            self.embeds[chain_req] = set()
        self.embeds[chain_req].add(i)

    def evict(self, chain_req):
        if chain_req in self.embeds:
            del self.embeds[chain_req]

    def add_dl(self, t, r):
        if t not in self.dl:
            self.dl[t] = 0
        self.dl[t] = self.dl[t] + r

    def rm_dl(self, t, r):
        if t not in self.dl:
            return
        self.dl[t] = self.dl[t] - r

    def __str__(self):
        return "{},{},{}".format(self.type, self.e1.id, self.e2.id)

    def __repr__(self):
        return self.__str__()


class Node:
    def __init__(self, t, loc, id):
        self.id = id
        self.type = t
        self.loc = loc
        self.sharing = True
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
        self.layers = dict()
        self.embeds = dict()
        self.q_agent = QLearn()
        self.s1 = None
        self.s1_extra = 0
        self.s2 = None

    def make_state(self):
        in_use = set()
        unused = set()
        for l in self.layers:
            if len(self.layers[l].chain_users) > 0:
                in_use.add(l)
            else:
                unused.add(l)
        return in_use, unused

    def make_s1(self):
        self.s1 = self.make_state()
        self.s1_extra, _ = self.get_all_unused()

    def make_s2(self):
        self.s2 = self.make_state()

    def get_local_reused(self):
        saved = 0
        for l in self.s1[1]: # unused at s1
            if l in self.s2[0]: # become inuse at s2
                saved = saved + self.layers[l].size
        return saved

    def get_local_kept(self):
        kept = set()
        vol = 0
        for l in self.s1[1]:  # unused at s1
            if l in self.s2[0] or l in self.s2[1]: # l is not in s2
                kept.add(l)
                vol = vol + self.layers[l].size
        return vol, kept

    def empty_storage_random(self, t):
        to_del = -1 * self.disk_avail(t)
        deleted = 0
        to_del_layer = list(self.get_unused_for_del(to_del))
        shuffle(to_del_layer)
        for l in to_del_layer:
            deleted = deleted + self.layers[l].size
            del self.layers[l]
        return deleted

    def empty_storage(self, t):
        to_del = -1 * self.disk_avail(t)
        deleted = 0
        if not self.q_agent.has_action(self.s1):
            deleted = self.empty_storage_random(t)
        else:
            will_remain = self.s1_extra - to_del
            to_keep = self.q_agent.get_action(self.s1, will_remain)
            if to_keep is None:
                deleted = self.empty_storage_random(t)
            else:
                will_be_deleted = set()
                will_be_deleted_size = 0
                for l in self.layers:
                    if len(self.layers[l].chain_users) == 0:
                        if l not in to_keep:
                            will_be_deleted.add(l)
                            will_be_deleted_size = will_be_deleted_size + self.layers[l].size
                            if will_be_deleted_size >= to_del:
                                break
                for l in will_be_deleted:
                    deleted = deleted + self.layers[l].size
                    del self.layers[l]
        # print("Deleted: {}".format(deleted))

    def reset(self):
        self.layers = dict()
        self.embeds = dict()
        self.q_agent = QLearn()

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
            if self.layer_inuse(my_layer):
                if t >= self.layers[my_layer].dl_start:
                    u = u + self.layers[my_layer].size
        return self.disk - u

    def get_all_unused(self):
        vol = 0
        unused = set()
        for my_layer in self.layers:
            if not self.layer_inuse(my_layer):
                unused.add(my_layer)
                vol = vol + self.layers[my_layer].size
        return vol, unused

    def get_unused_for_del(self, max_del):
        unused = set()
        for my_layer in self.layers:
            if not self.layer_inuse(my_layer):
                unused.add(my_layer)
                max_del = max_del - self.layers[my_layer].size
            if max_del <= 0:
                break
        return unused

    def disk_avail_ratio(self, t):
        a = self.disk_avail(t)
        return a / self.disk

    def layer_avail(self, r, t, chain_req=None):
        if self.type[0] == "b":
            return False
        if self.type[0] == "c":
            return True
        if self.sharing or chain_req is None:
            if r not in self.layers:
                return False
            return t >= self.layers[r].avail_from
        else:
            if (r, chain_req) not in self.layers:
                return False
            return t >= self.layers[(r, chain_req)].avail_from

    def layer_inuse(self, r):
        return len(self.layers[r].chain_users) > 0

    def embed(self, chain_req, i):
        if chain_req not in self.embeds:
            self.embeds[chain_req] = set()
        self.embeds[chain_req].add(i)

    def add_layer(self, R, chain_req):
        if self.sharing:
            for r in R:
                if r in self.layers:
                    self.layers[r].add_user(chain_req)
                else:
                    self.layers[r] = MyLayer(r, R[r], chain_req.arrival_time, chain_req.tau1)
                    self.layers[r].add_user(chain_req)
        else:
            for r in R:
                self.layers[(r, chain_req)] = MyLayer(r, R[r], chain_req.arrival_time, chain_req.tau1)
                self.layers[(r, chain_req)].add_user(chain_req)

    def evict(self, chain_req):
        if chain_req in self.embeds:
            del self.embeds[chain_req]
        for l in self.layers:
            self.layers[l].remove_user(chain_req)


class NetGenerator:
    def __init__(self):
        base_station_loc = [(0, 6), (3, 6), (6, 6), (0, 3), (0, 0), (3, 0), (6, 0)]
        base_station_loc = [(0, 0)]
        cloud_loc = (10, 3)
        self.g = nx.MultiDiGraph()
        for n in range(len(base_station_loc)):
            n_id = "b{}".format(n)
            nd = Node("base-station", base_station_loc[n], n_id)
            self.g.add_node(n_id, nd=nd)
        self.e_node_num = 1
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
        # connect cloud to nearest edge server
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
