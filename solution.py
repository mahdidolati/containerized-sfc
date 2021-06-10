import numpy as np


def usable_node(my_net, s, c, chain_req, i, t):
    if my_net.g.nodes[c]["nd"].cpu_avail(t) < chain_req.cpu_req(i):
        return False
    if my_net.g.nodes[c]["nd"].ram_avail(t) < chain_req.ramsk_req(i):
        return False
    R = dict()
    for r in chain_req.vnfs[i].layers:
        if r not in my_net.g.nodes[c]["nd"].layers:
            R[r] = my_net.g.nodes[c]["nd"].layers[r]
    d = 0
    for r in R:
        d = d + R[r]
    if my_net.g.nodes[c]["nd"].disk_avail(t) < d:
        return False
    if d > 0:
        path_bw, path_delay, links = my_net.get_biggest_path(c, "c")
        if chain_req.tau1 - t < np.ceil(d / path_bw) + 1:
            return False
        dl_rate = d / (chain_req.tau1 - t)
        for l in links:
            for tt in range(t, chain_req.tau1+1):
                l.set_dl(tt, dl_rate)

    if i > 0:
        path_bw, path_delay, links = my_net.get_biggest_path(s, c, t)
        if path_bw < chain_req.vnf_in_rate(i):
            return False
    return True


def solve(my_net, chain_req, t):
    delay_budge = chain_req.max_delay
    prev = None
    cur = my_net.get_closest(chain_req.entry_point)
    for i in range(len(chain_req.vnfs)):
        N1 = set()
        N2 = set()
        C = set()
        for c in N1:
            if usable_node(my_net, prev, c, chain_req, i, t):
                C.add(c)
