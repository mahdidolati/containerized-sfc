import gurobipy as gp
from gurobipy import GRB
from constants import Const


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


def solve_optimal(my_net, vnfs, R, Rvol, reqs):
    T = get_last_t(reqs) + 1
    I_len = get_max_sfc(reqs)
    B = my_net.get_all_base_stations()
    E = my_net.get_all_edge_nodes()
    N = len(E) + 1
    Lw, Lm, L_iii = my_net.get_link_sets()
    L = Lw + Lm
    L_len = len(L)
    cloud_node = "c"

    R_id = dict()
    r_idx = 0
    for r in R:
        R_id[r] = r_idx
        r_idx = r_idx + 1

    N_id = dict()
    n_idx = 0
    for e in E:
        N_id[e] = n_idx
        n_idx = n_idx + 1
    N_id[cloud_node] = n_idx
    n_idx = n_idx + 1
    for b in B:
        N_id[b] = n_idx
        n_idx = n_idx + 1

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
    # BINARY
    Gamma_var = m.addVars(len(R), T, len(E), vtype=GRB.BINARY, name="Gamma")
    Psi_var = m.addVars(len(R), T, len(E), vtype=GRB.BINARY, name="Psi")
    z_var = m.addVars(len(R), T, len(E), vtype=GRB.BINARY, name="z")
    y_var = m.addVars(len(R), T, len(E), vtype=GRB.BINARY, name="y")
    w_var = m.addVars(L_len, T, len(R), len(E), vtype=GRB.BINARY, name="w")
    v_var = m.addVars(N, T, len(reqs), I_len, vtype=GRB.BINARY, name="v")
    a_var = m.addVars(len(reqs), vtype=GRB.BINARY, name="a")
    q_var = m.addVars(L_len, T, len(reqs), I_len, vtype=GRB.BINARY, name="q")
    # CONTINUOUS
    G_var = m.addVars(len(R), T, len(E), vtype=GRB.CONTINUOUS, lb=0, name="G")
    g_var = m.addVars(len(R), T, len(E), vtype=GRB.CONTINUOUS, lb=0, name="g")
    wg_var = m.addVars(L_len, len(R), T, len(E), vtype=GRB.CONTINUOUS, name="wg")

    m.addConstrs(
        (
            Gamma_var[r, 0, e] == z_var[r, 0, e]
            for r in range(len(R))
            for e in range(len(E))
        ), name="dl_indicator_0"
    )

    m.addConstrs(
        (
            Gamma_var[r, t, e] == Gamma_var[r, t-1, e] + z_var[r, t, e] - y_var[r, t, e]
            for r in range(len(R))
            for e in range(len(E))
            for t in range(1, T)
        ), name="dl_indicator_1"
    )

    m.addConstrs(
        (
            gp.quicksum(
                w_var[l, t, r, e]
                for l in adj_out[N_id[cloud_node]]
            ) == Gamma_var[r, t, e]
            for r in range(len(R))
            for e in range(len(E))
            for t in range(T)
        ), name="dl_path_cloud"
    )

    m.addConstrs(
        (
            gp.quicksum(
                w_var[l, t, r, e]
                for l in adj_in[e]
            ) == Gamma_var[r, t, e]
            for r in range(len(R))
            for e in range(len(E))
            for t in range(T)
        ), name="dl_path_dst"
    )

    m.addConstrs(
        (
            gp.quicksum(
                w_var[l, t, r, e]
                for l in adj_out[ee]
            ) - gp.quicksum(
                w_var[l, t, r, e]
                for l in adj_in[ee]
            ) == 0
            for r in range(len(R))
            for e in range(len(E))
            for ee in range(len(E))
            for t in range(T)
            if e != ee
        ), name="dl_path_middle"
    )

    m.addConstrs(
        (
            G_var[r, t, e] <= Gamma_var[r, t, e] * Const.LAYER_SIZE[1]
            for r in range(len(R))
            for e in range(len(E))
            for t in range(T)
        ), name="dl_vol_0"
    )

    m.addConstrs(
        (
            G_var[r, 0, e] <= g_var[r, 0, e]
            for r in range(len(R))
            for e in range(len(E))
        ), name="dl_vol_1"
    )

    m.addConstrs(
        (
            G_var[r, t, e] <= G_var[r, t-1, e] + g_var[r, t, e]
            for r in range(len(R))
            for e in range(len(E))
            for t in range(1, T)
        ), name="dl_vol_2"
    )

    m.addConstrs(
        (
            gp.quicksum(
                Gamma_var[r, t, e] * Rvol[r]
                for r in range(len(R))
            ) <= my_net.g.nodes[E[e]]["nd"].disk
            for e in range(len(E))
            for t in range(T)
        ), name="disk_limit"
    )

    m.addConstrs(
        (
            Psi_var[r, t, e] * Rvol[r] <= G_var[r, t, e]
            for e in range(len(E))
            for r in range(len(R))
            for t in range(T)
        ), name="layer_avail"
    )

    m.addConstrs(
        (
            gp.quicksum(
                v_var[n, t, u, i]
                for n in range(N)
            ) == a_var[u]
            for u in range(len(reqs))
            for t in range(reqs[u].tau1, reqs[u].tau2+1)
            for i in range(len(reqs[u].vnfs))
        ), name="admit_placement"
    )

    m.addConstrs(
        (
            v_var[e, t, u, i] <= Psi_var[R_id[r], t, e]
            for u in range(len(reqs))
            for t in range(reqs[u].tau1, reqs[u].tau2 + 1)
            for e in range(len(E))
            for i in range(len(reqs[u].vnfs))
            for r in reqs[u].vnfs[i].layers
        ), name="vnf_layer"
    )

    m.addConstrs(
        (
            gp.quicksum(
                v_var[e, t, u, i] * reqs[u].vnfs[i].cpu * reqs[u].vnf_in_rate(i)
                for u in range(len(reqs))
                for i in range(len(reqs[u].vnfs))
            ) <= my_net.g.nodes[E[e]]["nd"].cpu
            for e in range(len(E))
            for t in range(T)
        ), name="cpu_limit"
    )

    m.addConstrs(
        (
            gp.quicksum(
                v_var[e, t, u, i] * reqs[u].vnfs[i].ram * reqs[u].vnf_in_rate(i)
                for u in range(len(reqs))
                for i in range(len(reqs[u].vnfs))
            ) <= my_net.g.nodes[E[e]]["nd"].ram
            for e in range(len(E))
            for t in range(T)
        ), name="ram_limit"
    )

    m.addConstrs(
        (
            gp.quicksum(
                q_var[l, t, u, 0]
                for l in adj_out[N_id[reqs[u].entry_point]]
            ) == a_var[u]
            for u in range(len(reqs))
            for t in range(reqs[u].tau1, reqs[u].tau2 + 1)
        ), name="entry_out"
    )

    m.addConstrs(
        (
            gp.quicksum(
                q_var[l, t, u, 0]
                for l in adj_in[n]
            ) == v_var[n, t, u, 0]
            for n in range(N)
            for u in range(len(reqs))
            for t in range(reqs[u].tau1, reqs[u].tau2 + 1)
        ), name="first_vnf_in"
    )

    m.addConstrs(
        (
            gp.quicksum(
                q_var[l, t, u, i+1]
                for l in adj_out[n]
            ) - gp.quicksum(
                q_var[l, t, u, i+1]
                for l in adj_in[n]
            ) == v_var[n, t, u, i] - v_var[n, t, u, i+1]
            for n in range(N)
            for u in range(len(reqs))
            for i in range(len(reqs[u].vnfs) - 1)
            for t in range(reqs[u].tau1, reqs[u].tau2 + 1)
        ), name="chaining"
    )

    max_dl_rate = max(Const.LINK_BW[1], Const.MM_BW[1])

    m.addConstrs(
        (
            wg_var[l, r, t, e] <= w_var[l, t, r, e] * max_dl_rate
            for l in range(len(L))
            for r in range(len(R))
            for t in range(T)
            for e in range(len(E))
        ), name="lin_0"
    )

    m.addConstrs(
        (
            wg_var[l, r, t, e] <= g_var[r, t, e]
            for l in range(len(L))
            for r in range(len(R))
            for t in range(T)
            for e in range(len(E))
        ), name="lin_1"
    )

    m.addConstrs(
        (
            wg_var[l, r, t, e] >= g_var[r, t, e] - (1 - w_var[l, t, r, e]) * max_dl_rate
            for l in range(len(L))
            for r in range(len(R))
            for t in range(T)
            for e in range(len(E))
        ), name="lin_2"
    )

    m.addConstrs(
        (
            gp.quicksum(
                wg_var[l, r, t, e]
                for r in range(len(R))
                for e in range(len(E))
            ) + gp.quicksum(
                q_var[l, t, u, i] * reqs[u].vnf_in_rate(i)
                for u in range(len(reqs))
                for i in range(len(reqs[u].vnfs))
            ) <= my_net.g[Lw[l][0]][Lw[l][1]][Lw[l][2]]["li"].bw
            for l in range(len(Lw))
            for t in range(T)
        ), name="bw_wired"
    )

    m.addConstrs(
        (
            gp.quicksum(
                wg_var[l, r, t, e]
                for l in adj_out[e]
                for r in range(len(R))
                if l >= len(Lw)
            ) + gp.quicksum(
                q_var[l, t, u, i] * reqs[u].vnf_in_rate(i)
                for l in adj_out[e]
                for u in range(len(reqs))
                for i in range(len(reqs[u].vnfs))
                if l >= len(Lw)
            ) <= my_net.g.nodes[E[e]]["nd"].mm_bw_tx
            for e in range(len(E))
            for t in range(T)
        ), name="bw_mm"
    )

    m.addConstrs(
        (
            gp.quicksum(
                q_var[l, t, u, i] * my_net.g[L[l][0]][L[l][1]][L[l][2]]["li"].delay
                for l in range(len(L))
                for i in range(len(reqs[u].vnfs))
            ) <= reqs[u].max_delay
            for u in range(len(reqs))
            for t in range(T)
        ), name="delay"
    )

    m.setObjective(
        gp.quicksum(
            a_var[u]
            for u in range(len(reqs))
        ),
        GRB.MAXIMIZE
    )

    m.setParam("Threads", 6)
    # m.setParam("TIME_LIMIT", 500)
    m.optimize()
    # m.write("out.lp")

    if m.status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write("model.ilp")
        return 0, 0

    DEBUG_FLAG = False
    tol_val = 0.0001
    if DEBUG_FLAG:
        for u in range(len(reqs)):
            if m.getVarByName("a[{}]".format(u)).x < tol_val:
                continue
            print("u: {} -- {}-{}".format(u, reqs[u].tau1, reqs[u].tau2))
            for i in range(len(reqs[u].vnfs)):
                locs = []
                for t in range(reqs[u].tau1, reqs[u].tau2 + 1):
                    i_loc = 0
                    dls = []
                    for n in range(N):
                        a = m.getVarByName("v[{},{},{},{}]".format(n, t, u, i)).x
                        if a > tol_val:
                            # print("\ti: {}, t: {}, n: {}".format(i, t, n))
                            locs.append(n)
                            i_loc = n
                    if i_loc < len(E):
                        for r in reqs[u].vnfs[i].layers:
                            a = m.getVarByName("G[{},{},{}]".format(R_id[r], t, i_loc)).x
                            if a + tol_val < Rvol[r]:
                                print("\tG(r: {}, t: {}, n: {}) = {} vs. {}".format(r, t, i_loc, a, Rvol[r]))
                                # dls.append(a)
                print("\tlocations: {}".format(locs))
                # print("\tdls: {}".format(dls))


    dl_vol = 0
    for e in range(len(E)):
        for r in range(len(R)):
            for t in range(T):
                g1 = 0
                g2 = 1 if m.getVarByName("Gamma[{},{},{}]".format(r, t, e)).x > tol_val else 0
                if t > 0:
                    g1 = 1 if m.getVarByName("Gamma[{},{},{}]".format(r, t-1, e)).x > tol_val else 0
                if g2 == 1 and g1 == 0:
                    dl_vol = dl_vol + Rvol[r]

    # return 0, 0
    return m.objVal / len(reqs), dl_vol / m.objVal if m.objVal > 0 else 0
