import cvxpy as cp
import numpy as np
from constants import Const


def get_last_t(reqs):
    t_max = 0
    for r in reqs:
        t_max = max(r.tau2, t_max)
    return int(t_max)


def solve_optimal(my_net, vnfs, R, Rvol, reqs):
    T = get_last_t(reqs)
    Lw, Lm = my_net.get_link_sets()
    L = Lw + Lm
    L_len = len(L)
    B = my_net.get_all_base_stations()
    E = my_net.get_all_edge_nodes()
    N = len(E) + 1
    a_var = cp.Variable((len(reqs),), boolean=True)
    z_var = cp.Variable((len(E)*len(R)*T,), boolean=True)
    y_var = cp.Variable((len(E)*len(R)*T,), boolean=True)
    Psi_var = cp.Variable((len(E)*len(R)*T,), boolean=True)
    Gamma_var = cp.Variable((len(E)*len(R)*T,))
    w_var = cp.Variable((len(E)*L_len*len(R)*T,), boolean=True)
    G_var = cp.Variable((len(E) * len(R) * T,))
    g_var = cp.Variable((len(E) * len(R) * T,))
    Gamma_G1_var = cp.Variable((len(E) * len(R) * T,))
    Gamma_g_var = cp.Variable((len(E) * len(R) * T,))
    v_var = cp.Variable((len(reqs) * len(vnfs) * N * T,))

    constraints = []

    # Only download or delete
    constraints += [
        z_var + y_var <= np.ones((len(E)*len(R)*T,))
    ]
    ###

    # Gamma var: Download status of layers
    t_0_end_1 = []
    t_1_end = []
    t_0_only = []
    for e in range(len(E)):
        for r in range(len(R)):
            for t in range(T):
                if t != T-1:
                    t_0_end_1.append(e*(len(R)*(T-1))+r*(T-1)+t)
                if t != 0:
                    t_1_end.append(e*(len(R)*(T-1))+r*(T-1)+t)
                if t == 0:
                    t_0_only.append(e*(len(R)*(T-1))+r*(T-1)+t)

    constraints += [
        Gamma_var[t_1_end] - Gamma_var[t_0_end_1] == z_var[t_1_end] - y_var[t_1_end]
    ]

    constraints += [
        Gamma_var[t_0_only] == np.zeros((len(E) * len(R),))
    ]
    ###

    # w var: Download path of layers
    adj_in = dict()
    adj_out = dict()
    cloud_node = "c"
    adj_in[cloud_node] = np.zeros((L_len * len(R) * T,))
    adj_out[cloud_node] = np.zeros((L_len * len(R) * T,))
    for l in range(L_len):
        if L[l][1] == cloud_node:
            adj_in[cloud_node][l * (len(R) * T):(l + 1) * (len(R) * T)] = np.ones((len(R) * T,))
        if L[l][0] == cloud_node:
            adj_out[cloud_node][l * (len(R) * T):(l + 1) * (len(R) * T)] = np.ones((len(R) * T,))

    for e in range(len(E)):
        adj_in[e] = np.zeros((L_len * len(R) * T,))
        adj_out[e] = np.zeros((L_len * len(R) * T,))
        for l in range(L_len):
            if L[l][1] == E[e]:
                adj_in[e][l * (len(R) * T):(l + 1) * (len(R) * T)] = np.ones((len(R) * T,))
            if L[l][0] == E[e]:
                adj_out[e][l * (len(R) * T):(l + 1) * (len(R) * T)] = np.ones((len(R) * T,))
        constraints += [
            w_var[e * (L_len * len(R) * T):(e + 1) * (L_len * len(R) * T)] @ adj_in[e]
            ==
            Gamma_var[e * (len(R) * T):(e + 1) * (len(R) * T)]
        ]
        constraints += [
            w_var[e * (L_len * len(R) * T):(e + 1) * (L_len * len(R) * T)] @ adj_out[cloud_node]
            ==
            Gamma_var[e * (len(R) * T):(e + 1) * (len(R) * T)]
        ]

    for e in range(len(E)):
        for ee in range(len(E)):
            if e != ee:
                constraints += [
                    w_var[e * (L_len * len(R) * T):(e + 1) * (L_len * len(R) * T)] @ adj_in[ee]
                    ==
                    w_var[e * (L_len * len(R) * T):(e + 1) * (L_len * len(R) * T)] @ adj_out[ee]
                ]
    ###

    # Linearize Gamma * g
    constraints += [
        Gamma_g_var <= Gamma_var * max(Const.LINK_BW[0], Const.MM_BW[1])
    ]

    constraints += [
        Gamma_g_var <= g_var
    ]
    ###

    # Linearize Gamma * G^t-1
    constraints += [
        Gamma_G1_var <= Gamma_var * Const.LAYER_SIZE[1]
    ]

    constraints += [
        Gamma_G1_var[t_1_end] <= G_var[t_0_end_1]
    ]

    constraints += [
        G_var[t_0_only] == np.zeros((len(E) * len(R),))
    ]
    ###

    # G var: Amount of download
    constraints += [
        G_var == Gamma_G1_var + Gamma_g_var
    ]
    ###

    # Disk capacity constraint
    agg = np.zeros((len(R)*T, T))
    for t in range(T):
        agg[t:len(R)*T:T, t] = np.ones((len(R),)) * Rvol
    for e in range(len(E)):
        constraints += [
            Gamma_var[e*len(R)*T:(e+1)*len(R)*T] @ agg <= np.ones((T,)) * my_net.g.nodes[E[e]]["nd"].disk
        ]
    ###

    # Psi var
    disk_req = np.ones((len(E)*len(R)*T,))
    for e in range(len(E)):
        for r in range(len(R)):
            disk_req[e * (len(R) * T) + r * T:e * (len(R) * T) + (r +1) * T] *= Rvol[r]
    constraints += [
        Psi_var * disk_req <= G_var
    ]
    ###

    # VNF layer relation
    layer_req = np.zeros((len(vnfs)*len(R)*len(E)*T,))
    for v in range(vnfs):
        for r in range(len(R)):
            if R[r] in vnfs[v].layers:
                layer_req[v*len(R)] = np.ones((len(E)*T,))

    ###

    print("model constructed...")
    objective = cp.Maximize(cp.sum(a_var))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CBC, verbose=True)

    print("Problem status {} with value {}".format(prob.status, prob.value))

    return 0, 0