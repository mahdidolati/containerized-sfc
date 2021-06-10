import numpy as np


def usable_node(my_net, s, c, chain_req, i, t):
    if my_net.g.nodes[c]["nd"].cpu_avail(t) < chain_req.cpu_req(i):
        return False
    if my_net.g.nodes[c]["nd"].ram_avail(t) < chain_req.ram_req(i):
        return False
    R = dict()
    for r in chain_req.vnfs[i].layers:
        if r not in my_net.g.nodes[c]["nd"].layers:
            R[r] = chain_req.vnfs[i].layers[r]
    d = 0
    for r in R:
        d = d + R[r]
    if my_net.g.nodes[c]["nd"].disk_avail() < d:
        return False
    dl_rate = d / (chain_req.tau1 - t)
    links = []
    if d > 0:
        path_bw, path_delay, links = my_net.get_biggest_path(c, "c")
        if chain_req.tau1 - t < np.ceil(d / path_bw) + 1:
            return False
        for l in links:
            for tt in range(t, chain_req.tau1+1):
                l.set_dl(tt, dl_rate)
    if i > 0:
        path_bw, path_delay, links = my_net.get_biggest_path(s, c, t)
        for l in links:
            for tt in range(t, chain_req.tau1+1):
                l.set_dl(tt, dl_rate)
        if path_bw < chain_req.vnf_in_rate(i):
            return False
    else:
        for l in links:
            for tt in range(t, chain_req.tau1+1):
                l.set_dl(tt, dl_rate)
    return True


def solve(my_net, chain_req, t):
    delay_budge = chain_req.max_delay
    prev = my_net.get_closest(chain_req.entry_point)
    for i in range(len(chain_req.vnfs)):
        N1 = my_net.g.nodes()
        C = set()
        for c in N1:
            if usable_node(my_net, prev, c, chain_req, i, t):
                C.add(c)
        if len(C) == 0:
            return False
        m = np.random.choice(C)
        my_net.g.nodes[m]["nd"].embed(chain_req, i)
        if prev != m:
            path_bw, path_delay, links = my_net.get_biggest_path(prev, m, t)
            for l in links:
                l.embed(chain_req, i)
        prev = m
    return True
