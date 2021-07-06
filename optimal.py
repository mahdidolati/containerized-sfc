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
    Lw, Lm = my_net.get_link_sets()
    B = my_net.get_all_base_stations()
    E = my_net.get_all_edge_nodes()
    a = cp.Variable(U, integer=True)
    z = cp.Variable((E, Rn, T), integer=True)
