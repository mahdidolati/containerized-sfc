import cvxpy as cp
import numpy as np


def get_last_t(reqs):
    t_max = 0
    for r in reqs:
        t_max = max(r.tau2, t_max)
    return int(t_max)


def get_max_sfc(reqs):
    l_max = 0
    for r in reqs:
        l_max = max(l_max, len(r.vnfs))
    return l_max


def solve_optimal(my_net, R, reqs):
    T = get_last_t(reqs)
    sfc_len = get_max_sfc(reqs)
    Lw, Lm = my_net.get_link_sets()
    L = Lw + Lm
    L_len = len(L)
    B = my_net.get_all_base_stations()
    E = my_net.get_all_edge_nodes()
    a_var = cp.Variable((len(reqs),), boolean=True)
    z_var = cp.Variable((len(E) * len(R) * T,), boolean=True)
    y_var = cp.Variable((len(E) * len(R) * T,), boolean=True)
    Psi_var = cp.Variable((len(E) * len(R) * T,), boolean=True)
    Gamma_var = cp.Variable((len(E) * len(R) * T,))
    w_var = cp.Variable((len(E) * L_len * len(R) * T,), boolean=True)
    G_var = cp.Variable((len(E) * len(R) * T,), nonneg=True)
    g_var = cp.Variable((len(E) * len(R) * T,), nonneg=True)
    v_var = cp.Variable((len(reqs) * sfc_len * T * len(E),), boolean=True)

    constraints = []

    # download or remove, not at the same time
    constraints += [
        z_var + y_var <= np.ones((len(E) * len(R) * T,))
    ]

    # download status Gamma
    t_0_end_1 = []
    t_1_end = []
    t_0 = []
    t_s = dict()
    for e in range(len(E)):
        for r in range(len(R)):
            for t in range(T):
                if t not in t_s:
                    t_s[t] = []
                if t != T - 1:
                    t_0_end_1.append(e * (len(R) * T) + r * T + t)
                if t != 0:
                    t_1_end.append(e * (len(R) * T) + r * T + t)
                t_s[t].append(e * (len(R) * T) + r * T + t)

    # DL at time 0
    constraints += [
        Gamma_var[t_s[0]] == z_var[t_s[0]] - y_var[t_s[0]]
    ]

    # layer DL indicator
    constraints += [
        Gamma_var[t_1_end] - Gamma_var[t_0_end_1] == z_var[t_1_end] - y_var[t_1_end]
    ]

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

    # no download unless start
    constraints += [
        G_var <= Gamma_var
    ]

    # download at time=0
    constraints += [
        G_var[t_s[0]] <= g_var[t_s[0]]
    ]

    # download at time>0
    constraints += [
        G_var[t_1_end] <= G_var[t_0_end_1] + g_var[t_1_end]
    ]

    # disk capacity
    d_agg = np.zeros((len(E) * len(R), len(E)))
    disk_r = []
    for r in range(len(R)):
        disk_r.append(R[r])
    D_e = []
    for e in range(len(E)):
        D_e.append(my_net.g.nodes[E[e]]["nd"].disk)
    for e in range(len(E)):
        d_agg[e * len(R):(e + 1) * len(R), e] = disk_r
    for t in range(T):
        constraints += [
            Gamma_var[t_s[t]] @ d_agg <= D_e
        ]

    # available after download
    disk_r_all = np.ones((len(E) * len(R) * T,))
    for e in range(len(E)):
        for r in range(len(R)):
            for t in range(T):
                disk_r_all[e * (len(R) * T) + r * T + t] = R[r]
    constraints += [
        cp.multiply(Psi_var, disk_r_all) <= G_var
    ]

    for u in range(len(reqs)):
        for i in range(len(reqs[u].vnfs)):
            for t in range(reqs[u].tau1, reqs[u].tau2 + 1):
                constraints += [
                    cp.sum(v_var[u * sfc_len * T * len(E)
                                 + i * T * len(E)
                                 + t * len(E)
                                 :u * sfc_len * T * len(E)
                                 + i * T * len(E) + (t + 1) * len(E)]
                           ) == a_var[u]
                ]

    print("model constructed...")
    objective = cp.Maximize(cp.sum(a_var))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CBC, verbose=True)

    print("Problem status {} with value {}".format(prob.status, prob.value))

    return 0, 0
