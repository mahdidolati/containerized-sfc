import cvxpy as cp
import numpy as np


def get_last_t(reqs):
    t_max = 0
    for r in reqs:
        t_max = max(r.tau2, t_max)
    return t_max


def solve_optimal(my_net, R, reqs):
    U = len(reqs)
    Rn = len(R)
    T = get_last_t(reqs)
    wired_link, mm_links = my_net.get_link_sets()
    all_bases = my_net.get_all_base_stations()
    all_enodes = my_net.get_all_edge_nodes()
    a_var = dict()
    for u in range(U):
        a_var[u] = cp.Variable(integer=True)
    z_var = dict()
    for e in range(len(all_enodes)):
        for r in range(Rn):
            for t in range(T):
                z_var[(e,r,t)] = cp.Variable(integer=True)
    return 0, 0