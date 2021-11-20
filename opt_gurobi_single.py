import gurobipy as gp
from gurobipy import GRB
from constants import Const
from itertools import chain
from sfc import LayerDownload


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


def solve_single(my_net, R, Rvol, req):
    I_len = len(req.vnfs)
    B = my_net.get_all_base_stations()
    E = my_net.get_all_edge_nodes()
    N = len(E) + 1
    Lw, L_iii = my_net.get_link_sets()
    L = Lw
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

    pre_computed_paths = dict()
    for e in E:
        pre_computed_paths[N_id[e]] = my_net.pre_compute_paths(e, req.arrival_time)

    adj_in = dict()
    adj_out = dict()
    for l in range(L_len):
        if N_id[L[l][0]] not in adj_out:
            adj_out[N_id[L[l][0]]] = list()
        if N_id[L[l][1]] not in adj_in:
            adj_in[N_id[L[l][1]]] = list()
        adj_out[N_id[L[l][0]]].append(l)
        adj_in[N_id[L[l][1]]].append(l)

    missing_layers = dict()
    for e in range(len(E)):
        for i in range(len(req.vnfs)):
            R_ei, _ = my_net.get_missing_layers(E[e], req, i, req.tau1)
            missing_layers[e, i] = R_ei

    T1 = range(req.arrival_time, req.tau1)
    T2 = range(req.tau1, req.tau2 + 1)

    m = gp.Model("Model")
    # BINARY
    v_var = m.addVars(N, I_len, vtype=GRB.BINARY, name="v")
    q_var = m.addVars(L_len, I_len+1, vtype=GRB.BINARY, name="q")
    y_var = m.addVars(len(E), 2, len(R), vtype=GRB.BINARY, name="y")

    m.addConstrs(
        (
            gp.quicksum(
                v_var[n, i]
                for n in range(N)
            ) == 1
            for i in range(len(req.vnfs))
        ), name="placement_all"
    )

    m.addConstrs(
        (
            gp.quicksum(
                v_var[e, i] * req.vnfs[i].cpu * req.vnf_in_rate(i)
                for i in range(len(req.vnfs))
            ) <= my_net.g.nodes[E[e]]["nd"].cpu_avail(t)
            for e in range(len(E))
            for t in T2
        ), name="cpu_limit"
    )

    m.addConstrs(
        (
            gp.quicksum(
                v_var[e, i] * req.vnfs[i].ram * req.vnf_in_rate(i)
                for i in range(len(req.vnfs))
            ) <= my_net.g.nodes[E[e]]["nd"].ram_avail(t)
            for e in range(len(E))
            for t in T2
        ), name="ram_limit"
    )

    m.addConstrs(
        (
            gp.quicksum(
                q_var[l, i] * req.vnf_in_rate(i)
                for i in range(len(req.vnfs)+1)
            ) <= my_net.g[Lw[l][0]][Lw[l][1]][Lw[l][2]]["li"].bw_avail(t)
            for l in range(len(Lw))
            for t in T2
        ), name="bw_wired"
    )

    m.addConstrs(
        (
            gp.quicksum(
                q_var[l, i + 1]
                for l in adj_out[n]
            ) - gp.quicksum(
                q_var[l, i + 1]
                for l in adj_in[n]
            ) == v_var[n, i] - v_var[n, i + 1]
            for n in range(N)
            for i in range(len(req.vnfs)-1)
            if n in adj_in and n in adj_out
        ), name="chaining"
    )

    m.addConstr(
        gp.quicksum(
            q_var[l, 0]
            for l in adj_out[N_id[req.entry_point]]
        ) == 1,
        name="entry_out"
    )

    m.addConstrs(
        (
            gp.quicksum(
                q_var[l, 0]
                for l in adj_in[n]
            ) == v_var[n, 0]
            for n in range(N)
            if n in adj_in
        ), name="first_vnf_in"
    )

    m.addConstr(
        gp.quicksum(
            q_var[l, len(req.vnfs)]
            for l in adj_in[N_id[req.entry_point]]
        ) == 1,
        "entry_in"
    )

    m.addConstrs(
        (
            gp.quicksum(
                q_var[l, len(req.vnfs)]
                for l in adj_out[n]
            ) == v_var[n, len(req.vnfs)-1]
            for n in range(N)
            if n in adj_out
        ), name="last_vnf_out"
    )

    m.addConstr(
        gp.quicksum(
            q_var[l, i] * my_net.g[L[l][0]][L[l][1]][L[l][2]]["li"].delay
            for l in range(len(L))
            for i in range(len(req.vnfs))
        ) <= req.max_delay,
        name="delay"
    )

    m.addConstrs(
        (
            v_var[e, i] <= gp.quicksum(
                y_var[e, p, R_id[r]]
                for p in range(len(pre_computed_paths[e]))
            )
            for e in range(len(E))
            for i in range(len(req.vnfs))
            for r in missing_layers[(e,i)]
        ), name="choose_dl_path"
    )

    m.addConstrs(
        (
            gp.quicksum(
                y_var[e, p, r] * Rvol[r] / len(T1)
                for r in range(len(R))
                for e in range(len(E))
                for p in range(len(pre_computed_paths[e]))
                if Lw[l] in pre_computed_paths[e][p]
            ) <= my_net.g[Lw[l][0]][Lw[l][1]][Lw[l][2]]["li"].bw_avail(t)
            for l in range(len(Lw))
            for t in T1
        ), name="dl_bw_wired"
    )

    m.addConstrs(
        (
            gp.quicksum(
                y_var[e, p, r] * Rvol[r]
                for r in range(len(R))
                for p in range(len(pre_computed_paths[e]))
            ) <= my_net.g.nodes[E[e]]["nd"].disk_avail_no_cache(t)
            for e in range(len(E))
            for t in chain(T1, T2)
        ), name="disk_limit"
    )

    m.setObjective(
        gp.quicksum(
            q_var[l, i] * req.vnf_in_rate(i)
            for l in range(len(L))
            for i in range(len(req.vnfs))
        ) + gp.quicksum(
            q_var[l, len(req.vnfs)] * req.vnf_out_rate(len(req.vnfs)-1)
            for l in range(len(L))
        ),
        GRB.MINIMIZE
    )

    m.setParam("LogToConsole", False)
    m.setParam("Threads", 6)
    # m.setParam("TIME_LIMIT", 500)
    m.optimize()
    # m.write("out.lp")

    if m.status == GRB.INFEASIBLE:
        # m.computeIIS()
        # m.write("s_model.ilp")
        return False, None

    tol_val = 0.0001
    for i in range(len(req.vnfs)):
        for n in range(N):
            a = m.getVarByName("v[{},{}]".format(n, i)).x
            if abs(a - 1.0) < tol_val:
                n_name = E[n] if n < len(E) else cloud_node
                my_net.g.nodes[n_name]["nd"].embed(req, i)
                # print("Placed {} on {}".format(i, n_name))
                if n < len(E):
                    my_net.g.nodes[n_name]["nd"].add_layer(missing_layers[(n, i)], req)

    total_dl_vol = 0
    downloads = []
    for e in range(len(E)):
        for r in range(len(R)):
            for p in range(len(pre_computed_paths[e])):
                a = m.getVarByName("y[{},{},{}]".format(e, p, r)).x
                if abs(a - 1.0) < tol_val:
                    total_dl_vol = total_dl_vol + Rvol[r]
                    layer_download = LayerDownload()
                    downloads.append(layer_download)
                    for l in pre_computed_paths[e][p]:
                        l_obj = my_net.g[l[0]][l[1]][l[2]]["li"]
                        for tt in T1:
                            layer_download.add_data(tt, l_obj, Rvol[r] / len(T1))

    for l in range(len(L)):
        for i in range(len(req.vnfs) + 1):
            a = m.getVarByName("q[{},{}]".format(l, i)).x
            if abs(a - 1.0) < tol_val:
                l_obj = my_net.g[L[l][0]][L[l][1]][L[l][2]]["li"]
                l_obj.embed(req, i)
                # print("Link embed {} <--> {} towards {}".format(L[l][0], L[l][1], i))

    return True, total_dl_vol
