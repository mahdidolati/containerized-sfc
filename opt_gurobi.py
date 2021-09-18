import gurobipy as gp
from gurobipy import GRB
import numpy as np
from constants import Const


def get_last_t(reqs):
    t_max = 0
    for r in reqs:
        t_max = max(r.tau2, t_max)
    return int(t_max)


def solve_optimal(my_net, vnfs, R, Rvol, reqs):
    T = get_last_t(reqs)
    E = my_net.get_all_edge_nodes()
    N = len(E) + 1
    Lw, Lm = my_net.get_link_sets()
    L = Lw + Lm
    L_len = len(L)
    cloud_node = "c"

    N_id = dict()
    n_idx = 0
    for e in E:
        N_id[e] = n_idx
        n_idx = n_idx + 1
    N_id[cloud_node] = n_idx

    adj_in = dict()
    adj_out = dict()
    for l in range(L_len):
        if N_id[L[l][0]] not in adj_out:
            adj_out[N_id[L[l][0]]] = list()
        if N_id[L[l][1]] not in adj_in:
            adj_in[N_id[L[l][1]]] = list()
        adj_out[N_id[L[l][0]]].append(l)
        adj_in[N_id[L[l][1]]].append(l)

    m = gp.Model("Model")
    Gamma_var = m.addVars(len(R), T, len(E), vtype=GRB.BINARY, name="Gamma")
    G_var = m.addVars(len(R), T, len(E), vtype=GRB.BINARY, name="G")
    g_var = m.addVars(len(R), T, len(E), vtype=GRB.BINARY, name="g")
    z_var = m.addVars(len(R), T, len(E), vtype=GRB.BINARY, name="z")
    y_var = m.addVars(len(R), T, len(E), vtype=GRB.BINARY, name="y")
    w_var = m.addVars(L_len, T, len(R), len(E), vtype=GRB.BINARY, name="w")

    m.addConstrs(
        (
            Gamma_var[r, 0, e] == z_var[r, 0, e] - y_var[r, 0, e]
            for r in range(R)
            for e in range(E)
        ), name="dl_indicator_0"
    )

    m.addConstrs(
        (
            Gamma_var[r, t, e] - Gamma_var[r, t-1, e] == z_var[r, t, e] - y_var[r, t, e]
            for r in range(R)
            for e in range(E)
            for t in range(1, T)
        ), name="dl_indicator_1"
    )

    m.addConstrs(
        (
            quicksum(
                w_var[l, t, r, e]
                for l in adj_out[N_id[cloud_node]]
            ) == Gamma_var[r, t, e]
            for r in range(R)
            for e in range(E)
            for t in range(1, T)
        ), name="dl_path_cloud"
    )

    m.addConstrs(
        (
            quicksum(
                w_var[l, t, r, e]
                for l in adj_in[e]
            ) == Gamma_var[r, t, e]
            for r in range(R)
            for e in range(E)
            for t in range(T)
        ), name="dl_path_dst"
    )

    m.addConstrs(
        (
            quicksum(
                w_var[l, t, r, e]
                for l in adj_out[ee]
            ) - quicksum(
                w_var[l, t, r, e]
                for l in adj_in[ee]
            ) == 0
            for r in range(R)
            for e in range(E)
            for ee in range(E)
            for t in range(T)
            if e != ee
        ), name="dl_path_middle"
    )

    m.addConstrs(
        (
            G_var[r, t, e] <= Gamma_var[r, t, e]
            for r in range(R)
            for e in range(E)
            for t in range(T)
        ), name="dl_vol_0"
    )

    m.addConstrs(
        (
            G_var[r, t, e] <= Gamma_var[r, t-1, e] + g_var[r, t-1, e]
            for r in range(R)
            for e in range(E)
            for t in range(1, T)
        ), name="dl_vol_0"
    )





