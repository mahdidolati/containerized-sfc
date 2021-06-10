import numpy as np


def get_closest(g, n):
    bestDelay = np.infty
    bestNeighbor = None
    for m in g.neighbors(n):
        for j in g[n][m]:
            if bestNeighbor is None or bestDelay > g[m][n][j]["delay"]:
                bestDelay = g[m][n][j]["delay"]
                bestNeighbor = (n, j)
    return bestNeighbor


def usable_node(my_net, s, c, chain_req, i, t):
    if my_net.g.nodes[c]["nd"].avail_cpu() < chain_req.cpu_req(i):
        return False
    if my_net.g.nodes[c]["nd"].avail_ram() < chain_req.ramsk_req(i):
        return False
    R = dict()
    for r in chain_req.vnfs[i].layers:
        if r not in my_net.g.nodes[c]["nd"].layers:
            R[r] = my_net.g.nodes[c]["nd"].layers[r]
    d = 0
    for r in R:
        d = d + R[r]
    if my_net.g.nodes[c]["nd"].avail_ram() < d:
        return False
    if d > 0:
        bw, path = my_net.get_biggest_path(c, "c")
        if chain_req.tau1 - t < np.ceil(d / bw) + 1:
            return False
    if i > 0:
        bw, path = my_net.get_biggest_path(s, c)
        if bw < chain_req.vnf_in_rate(i):
            return False
    return True


def solve(my_net, chain_req, t):
    delay_budge = chain_req.max_delay
    prev = None
    cur = get_closest(my_net, chain_req.entry_point)
    for i in range(len(chain_req.vnfs)):
        N1 = set()
        N2 = set()
        C = set()
        for c in N1:
            if usable_node(my_net, prev, c, chain_req, i, t):
                C.add(c)
