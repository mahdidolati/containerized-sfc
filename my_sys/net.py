import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import heapq
from constants import Const
from sfc import LayerDownload
from q_learn import QLearn
from random import shuffle
from popularity_learn import PLearn


class MyLayer:
    def __init__(self, layer_no, size, dl_start, avail_from):
        self.layer_no = layer_no
        self.size = size
        self.dl_start = dl_start
        self.avail_from = avail_from
        self.last_used = avail_from
        self.chain_users = set()
        self.finalized = False

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
        self.paths_links = dict()
        self.paths_nodes = dict()
        for n1 in self.g.nodes():
            self.paths_links[n1] = dict()
            self.paths_nodes[n1] = dict()
            for n2 in self.g.nodes():
                if n1 != n2:
                    self.paths_links[n1][n2], self.paths_nodes[n1][n2] = self.pre_compute_paths(n1, n2)
        self.link_to_path = dict()
        for n1 in self.paths_links:
            for n2 in self.paths_links[n1]:
                for pth_id in range(len(self.paths_links[n1][n2])):
                    for ll in self.paths_links[n1][n2][pth_id]:
                        if ll not in self.link_to_path:
                            self.link_to_path[ll] = set()
                        self.link_to_path[ll].add((n1, n2, pth_id))

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
                cur_path.append(d)
                return -bw, cur_path, cur_links
            if n in visited:
                continue
            visited.add(n)
            for m in self.g.neighbors(n):
                if m not in visited and m not in cur_path:
                    cur_link = self.g[n][m]["li"]
                    if (n, m) not in excluded:
                        bw_avail = cur_link.bw_avail(t)
                        counter = counter + 1
                        new_path = list(cur_path)
                        new_path.append(n)
                        new_links = list(cur_links)
                        new_links.append((n, m))
                        heapq.heappush(h, (max(bw, -bw_avail),
                                           counter, m, new_path, new_links))
        return 0, [], []

    def pre_compute_paths(self, n1, n2, t=0):
        paths_link = []
        paths_nodes = []
        bb, pp, ll = self.get_a_path(n1, n2, t)
        paths_link.append(ll)
        paths_nodes.append(pp)
        for l in ll:
            bb2, pp2, ll2 = self.get_a_path(n1, n2, t, [l])
            if bb2 > 0:
                paths_link.append(ll2)
                paths_nodes.append(pp2)
        return paths_link, paths_nodes

    def get_path_min_bw(self, n1, n2, pth_id, T):
        min_bw = np.infty
        for ll in self.paths_links[n1][n2][pth_id]:
            bw_a = self.g[ll[0]][ll[1]]["li"].bw_min_avail(T)
            if bw_a < min_bw:
                min_bw = bw_a
        return min_bw

    def get_path_delay(self, n1, n2, pth_id):
        d = 0
        for ll in self.paths_links[n1][n2][pth_id]:
            d = d + self.g[ll[0]][ll[1]]["li"].delay
        return d

    def get_closest(self, n):
        bestDelay = np.infty
        bestNeighbor = None
        for m in self.g.neighbors(n):
            if bestNeighbor is None or bestDelay > self.g[m][n]["delay"]:
                bestDelay = self.g[m][n]["li"].delay
                bestNeighbor = (n, m)
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

    def get_net_util(self, t):
        e_cpu = dict()
        e_ram = dict()
        e_disk = dict()
        for n in self.get_all_edge_nodes():
            c = self.g.nodes[n]["nd"].cpu_util(t)
            r = self.g.nodes[n]["nd"].ram_util(t)
            d = self.g.nodes[n]["nd"].disk_util(t)
            e_cpu[n] = c
            e_ram[n] = r
            e_disk[n] = d
        return e_cpu, e_ram, e_disk

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
        d = 0
        for r in chain_req.vnfs[vnf_i].layers:
            if not self.g.nodes[server]["nd"].layer_avail(r, t) \
                    or not self.g.nodes[server]["nd"].layer_inuse(r):
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
            self.g[e[0]][e[1]]["li"].evict(chain_req)

    def get_link_sets(self):
        Lw = list()
        L_iii = dict()
        for e in self.g.edges():
            L_iii[self.g[e[0]][e[1]]["li"]] = (e[0], e[1])
            Lw.append((e[0], e[1]))
        return Lw, L_iii

    def get_all_base_stations(self):
        B = list()
        for n in self.g.nodes():
            if n[0] == "b":
                B.append(n)
        return B

    def get_total_out_bw(self, n, T):
        t_bw = 0
        for e in self.g.edges():
            if e[0] == n:
                t_bw = t_bw + self.g[e[0]][e[1]]["li"].bw_min_avail(T)
        return t_bw

    def get_total_in_bw(self, n, T):
        t_bw = 0
        for e in self.g.edges():
            if e[1] == n:
                t_bw = t_bw + self.g[e[0]][e[1]]["li"].bw_min_avail(T)
        return t_bw

    def reset(self):
        for n in self.g.nodes():
            self.g.nodes[n]["nd"].reset()
        for e in self.g.edges():
            self.g[e[0]][e[1]]["li"].reset()

    def print(self):
        for n in self.g.nodes():
            ss = self.g.nodes[n]["nd"].cpu
            rr = self.g.nodes[n]["nd"].ram
            dd = self.g.nodes[n]["nd"].disk
            print("Node: {}, cpu: {}, ram: {}, disk: {}".format(n, ss, rr, dd))
        for e in self.g.edges():
            tt = self.g[e[0]][e[1]]["li"].type
            bb = self.g[e[0]][e[1]]["li"].bw
            dd = self.g[e[0]][e[1]]["li"].delay
            print("Link({}): {} -- {}, bw: {}, dl: {}".format(tt, e[0], e[1], bb, dd))


class Link:
    def __init__(self, tp, s, d):
        self.type = tp
        self.e1 = s
        self.e2 = d
        if self.e1.id[0] == "c" or self.e2.id[0] == "c":
            self.bw = np.infty
            self.delay = 2000 * np.linalg.norm(self.e1.loc - self.e2.loc)
        else:
            self.delay = 10 * np.linalg.norm(self.e1.loc - self.e2.loc)
            self.bw = np.random.randint(*Const.LINK_BW)
        # print(self.delay)
        self.embeds = dict()
        self.dl = dict()

    def reset(self):
        self.embeds = dict()
        self.dl = dict()

    def bw_min_avail(self, T):
        min_bw = np.infty
        for t in T:
            bw_a = self.bw_avail(t)
            if bw_a < min_bw:
                min_bw = bw_a
        return min_bw

    def bw_avail(self, t):
        u = 0
        for r in self.embeds:
            if r.tau1 <= t <= r.tau2:
                for i in self.embeds[r]:
                    u = u + r.vnf_in_rate(i)
        if t in self.dl:
            u = u + self.dl[t]
        return self.bw - u

    def unembed(self, chain_req, i):
        if chain_req in self.embeds:
            self.embeds[chain_req].remove(i)

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
        self.p_agent = PLearn()

    def make_state(self):
        in_use = set()
        unused = set()
        for l in self.layers:
            if self.layer_inuse(l):
                in_use.add(l)
            else:
                unused.add(l)
        return in_use, unused

    def make_s1(self):
        self.s1 = self.make_state()
        self.s1_extra, _ = self.get_all_unused()

    def make_s2(self):
        self.s2 = self.make_state()

    # for reward
    def get_local_reused(self):
        saved = 0
        for l in self.s1[1]: # unused at s1
            if l in self.s2[0]: # become inuse at s2
                saved = saved + self.layers[l].size
        return saved

    # for action
    def get_local_kept(self):
        kept = set()
        vol = 0
        for l in self.s1[1]:  # unused at s1
            if l in self.s2[0] or l in self.s2[1]: # l is not in s2
                kept.add(l)
                vol = vol + self.layers[l].size
        return vol, kept

    def empty_storage_popularity(self, t):
        to_del = -1 * self.disk_avail(t)
        _, unused_layers = self.get_all_unused()
        to_del_set = self.p_agent.get_action(unused_layers, to_del, self.layers)
        for ll in to_del_set:
            del self.layers[ll]

    def empty_storage_random(self, t):
        to_del = -1 * self.disk_avail(t)
        _, unused_layers = self.get_all_unused()
        unused_layers = list(unused_layers)
        shuffle(unused_layers)
        for l in unused_layers:
            to_del = to_del - self.layers[l].size
            del self.layers[l]
            if to_del <= 0:
                break

    def empty_storage(self, t):
        to_del = -1 * self.disk_avail(t)
        deleted = 0
        if not self.q_agent.has_action(self.s1):
            self.empty_storage_random(t)
        else:
            will_remain = self.s1_extra - to_del
            to_keep = self.q_agent.get_action(self.s1, will_remain)
            if to_keep is None:
                self.empty_storage_random(t)
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

    def reset(self):
        self.layers = dict()
        self.embeds = dict()
        # self.q_agent = QLearn()
        self.p_agent = PLearn()

    def cpu_min_avail(self, T):
        min_cpu = np.infty
        for t in T:
            cpu_a = self.cpu_avail(t)
            if cpu_a < min_cpu:
                min_cpu = cpu_a
        # print("In period {}, Cpu available is: {}".format(T, min_cpu))
        return min_cpu

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

    def cpu_util(self, t):
        cpu_a = self.cpu_avail(t)
        return (self.cpu - cpu_a) / self.cpu

    def ram_min_avail(self, T):
        min_ram = np.infty
        for t in T:
            ram_a = self.ram_avail(t)
            if ram_a < min_ram:
                min_ram = ram_a
        # print("In period {}, Ram available is: {}".format(T, min_ram))
        return min_ram

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

    def ram_util(self, t):
        ram_a = self.ram_avail(t)
        return (self.ram - ram_a) / self.ram

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

    def disk_util(self, t):
        disk_a = self.disk_avail(t)
        return (self.disk - disk_a) / self.disk

    def disk_min_avail_no_cache(self, T):
        min_disk = np.infty
        for t in T:
            disk_a = self.disk_avail_no_cache(t)
            if disk_a < min_disk:
                min_disk = disk_a
        # print("In period {}, Disk available is: {}".format(T, min_disk))
        return min_disk

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

    def get_all_inuse(self):
        inuse = set()
        for my_layer in self.layers:
            if self.layer_inuse(my_layer):
                inuse.add(my_layer)
        return inuse

    def get_unused_for_del(self, max_del):
        unused = set()
        for my_layer in self.layers:
            if not self.layer_inuse(my_layer):
                unused.add(my_layer)
                max_del = max_del - self.layers[my_layer].size
            if max_del <= 0:
                break
        return unused

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

    def finalize_layer(self):
        for ll in self.layers:
            self.layers[ll].finalized = True

    def unembed(self, chain_req, i):
        if chain_req in self.embeds:
            if i in self.embeds[chain_req]:
                self.embeds[chain_req].remove(i)
            rm_usr = set()
            for ll in chain_req.vnfs[i].layers:
                to_be_rm = True
                for ii in self.embeds[chain_req]:
                    if ll in chain_req.vnfs[ii].layers:
                        to_be_rm = False
                        break
                if to_be_rm:
                    rm_usr.add(ll)
            for ll in rm_usr:
                if ll in self.layers:
                    self.layers[ll].remove_user(chain_req)
                    if len(self.layers[ll].chain_users) == 0 and not self.layers[ll].finalized:
                        del self.layers[ll]

    def embed(self, chain_req, i):
        if chain_req not in self.embeds:
            self.embeds[chain_req] = set()
        self.embeds[chain_req].add(i)

    def evict(self, chain_req):
        if chain_req in self.embeds:
            del self.embeds[chain_req]
        for l in self.layers:
            self.layers[l].remove_user(chain_req)


class NetGenerator:
    def __init__(self):
        base_station_loc = [(0, 6), (3, 6), (6, 6), (0, 3), (0, 0), (3, 0), (6, 0)]
        base_station_loc = [(0, 6), (3, 6), (6, 6), (0, 3), (0, 0)]
        self.e_node_num = 7
        cloud_loc = (50, 3)
        self.g = nx.DiGraph()
        for n in range(len(base_station_loc)):
            n_id = "b{}".format(n)
            nd = Node("base-station", base_station_loc[n], n_id)
            self.g.add_node(n_id, nd=nd)
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
        for n in range(self.e_node_num-1):
            e1 = "e{}".format(n)
            e2 = "e{}".format(n+1)
            li1 = Link("wired", self.g.nodes[e1]["nd"], self.g.nodes[e2]["nd"])
            li2 = Link("wired", self.g.nodes[e2]["nd"], self.g.nodes[e1]["nd"])
            self.g.add_edge(e1, e2, li=li1)
            self.g.add_edge(e2, e1, li=li2)
        #
        for l in combinations(range(self.e_node_num), 2):
            if l[0] + 1 != l[1]:
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
        #     line_t = 'r-'
        #     if self.g[e[0]][e[1]]["li"].type == "mmWave":
        #         line_t = 'b-'
        #     ax.plot([self.g.nodes[e[0]]["nd"].loc[0], self.g.nodes[e[1]]["nd"].loc[0]],
        #             [self.g.nodes[e[0]]["nd"].loc[1], self.g.nodes[e[1]]["nd"].loc[1]], line_t)
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
