import gurobipy as gp
from gurobipy import GRB
from constants import Const
from itertools import chain
from sfc import LayerDownload
import numpy as np


def get_T(reqs):
    t1 = np.infty
    t2 = 0
    for r in reqs:
        t1 = min(r.arrival_time, t1)
        t2 = max(r.tau2, t2)
    return range(t1, t2+1)


def solve_batch_opt(reqs, my_net, R, Rvol):
    a_reqs = list()
    for ii in range(len(reqs)):
        a_reqs.append(reqs[ii])
        m, v_var, q_var, w_var, r_var, T_all, R_id, E_id, Ec_id, N_map, N_map_inv, cloud_node = get_ilp(a_reqs, my_net, R,
                                                                                                    Rvol)
        m.setParam("LogToConsole", False)
        m.setParam("Threads", 6)
        m.setParam("TIME_LIMIT", 30)
        m.optimize()
        # m.write("out.lp")

        if m.status == GRB.INFEASIBLE:
            # m.computeIIS()
            # m.write("s_model.ilp")
            # return False, 1, 0
            print("rejected one!")
            a_reqs = a_reqs[:-1]

    print(m.objVal)
    return 1.0 * (len(a_reqs))/len(reqs), 1, 1, m.objVal


def get_ilp(reqs, my_net, R, Rvol):
    T_all = get_T(reqs)
    B = my_net.get_all_base_stations()
    E = my_net.get_all_edge_nodes()
    cloud_node = "c"
    Ec = E + [cloud_node]
    N = Ec + B

    L, L_iii = my_net.get_link_sets()

    R_id = dict()
    r_idx = 0
    for r in R:
        R_id[r] = r_idx
        r_idx = r_idx + 1

    E_id = list()
    Ec_id = list()
    N_map = dict()
    N_map_inv = dict()
    n_idx = 0
    for n in N:
        if n in E:
            E_id.append(n_idx)
            Ec_id.append(n_idx)
        if n == cloud_node:
            Ec_id.append(n_idx)
        N_map[n] = n_idx
        N_map_inv[n_idx] = n
        n_idx = n_idx + 1

    m = gp.Model("Model")
    # BINARY
    v_var = dict()
    q_var = dict()
    w_var = dict()
    for req_id in range(len(reqs)):
        if req_id not in q_var:
            q_var[req_id] = dict()
        if req_id not in w_var:
            w_var[req_id] = dict()
        v_var[req_id] = m.addVars(Ec_id, len(reqs[req_id].vnfs), vtype=GRB.BINARY, name="v,{},".format(req_id))
        for n1 in my_net.paths_links:
            if n1 not in q_var[req_id]:
                q_var[req_id][N_map[n1]] = dict()
            if n1 not in w_var[req_id]:
                w_var[req_id][N_map[n1]] = dict()
            for n2 in my_net.paths_links[n1]:
                if len(my_net.paths_links[n1][n2]) > 0:
                    # print("{} -- {}: {} , {}".format(n1, n2, len(my_net.paths_links[n1][n2]), len(reqs[req_id].vnfs)+1))
                    q_var[req_id][N_map[n1]][N_map[n2]] = m.addVars(len(my_net.paths_links[n1][n2]), len(reqs[req_id].vnfs)+1,
                                                      vtype=GRB.BINARY, name="q,{},{},{},".format(req_id,n1,n2))
                    w_var[req_id][N_map[n1]][N_map[n2]] = m.addVars(len(my_net.paths_links[n1][n2]), len(R),
                                                      vtype=GRB.BINARY, name="w,{},{},{},".format(req_id,n1,n2))
    r_var = m.addVars(E_id, len(R), T_all, vtype=GRB.BINARY, name="r")

    m.addConstrs(
        (
            gp.quicksum(
                v_var[req_id][n, i]
                for n in Ec_id
            ) == 1
            for req_id in range(len(reqs))
            for i in range(len(reqs[req_id].vnfs))
        ), name="placement_all"
    )

    m.addConstrs(
        (
            gp.quicksum(
                v_var[req_id][e, i] * reqs[req_id].cpu_req(i)
                for req_id in range(len(reqs))
                if t in reqs[req_id].T2
                for i in range(len(reqs[req_id].vnfs))
            ) <= my_net.g.nodes[N_map_inv[e]]["nd"].cpu_avail(t)
            for e in E_id
            for t in T_all
        ), name="cpu_limit"
    )

    m.addConstrs(
        (
            gp.quicksum(
                v_var[req_id][e, i] * reqs[req_id].ram_req(i)
                for req_id in range(len(reqs))
                if t in reqs[req_id].T2
                for i in range(len(reqs[req_id].vnfs))
            ) <= my_net.g.nodes[N_map_inv[e]]["nd"].ram_avail(t)
            for e in E_id
            for t in T_all
        ), name="ram_limit"
    )

    m.addConstrs(
        (
            v_var[req_id][e, i] <= r_var[e, R_id[r], t]
            for e in E_id
            for req_id in range(len(reqs))
            for i in range(len(reqs[req_id].vnfs))
            for r in reqs[req_id].vnfs[i].layers
            for t in reqs[req_id].T2
        ), name="layer_prereq"
    )

    if 0 in T_all:
        m.addConstrs(
            (
                r_var[e, R_id[r], 0] == 0
                for e in E_id
                for r in R
            ), name="layer_0"
        )

    for e in E_id:
        for r in R:
            for t in T_all:
                if t == 0:
                    continue
                if my_net.g.nodes[N_map_inv[e]]["nd"].layer_avail(r, t):
                    if my_net.g.nodes[N_map_inv[e]]["nd"].layer_inuse(r):
                        r_var[e, R_id[r], t].lb = 1.0
                    else:
                        pass # may assume or not with no problem
                else:
                    if t-1 not in T_all:
                        m.addConstr(
                            r_var[e, R_id[r], t] <= gp.quicksum(
                                w_var[req_id][e][N_map[cloud_node]][pth_id, R_id[r]]
                                for req_id in range(len(reqs))
                                if t in reqs[req_id].T2
                                for pth_id in range(len(my_net.paths_links[N_map_inv[e]][cloud_node]))
                            ), name="layer_download,{},{},{}".format(e, r, t)
                        )
                    else:
                        m.addConstr(
                            r_var[e, R_id[r], t] <= r_var[e, R_id[r], t - 1] + gp.quicksum(
                                w_var[req_id][e][N_map[cloud_node]][pth_id, R_id[r]]
                                for req_id in range(len(reqs))
                                if t in reqs[req_id].T2
                                for pth_id in range(len(my_net.paths_links[N_map_inv[e]][cloud_node]))
                            ), name="layer_download,{},{},{}".format(e, r, t)
                        )

    m.addConstrs(
        (
            gp.quicksum(
                w_var[req_id][e][N_map[cloud_node]][pth_id, R_id[r]]
                for pth_id in range(len(my_net.paths_links[N_map_inv[e]][cloud_node]))
            ) <= 1
            for req_id in range(len(reqs))
            for e in E_id
            for r in R
        ), name="download_path_1"
    )

    m.addConstrs(
        (
            gp.quicksum(
                r_var[e, R_id[r], t] * Rvol[R_id[r]]
                for r in R
            ) <= my_net.g.nodes[N_map_inv[e]]["nd"].disk
            for e in E_id
            for t in T_all
        ), name="disk"
    )

    m.addConstrs(
        (
            gp.quicksum(
                q_var[req_id][N_map[reqs[req_id].entry_point]][n][pth_id, 0]
                for pth_id in range(len(my_net.paths_links[reqs[req_id].entry_point][N_map_inv[n]]))
            ) == v_var[req_id][n, 0]
            for req_id in range(len(reqs))
            for n in Ec_id
        ), name="entry_in"
    )

    # for req_id in range(len(reqs)):
    #     for n in Ec_id:
    #         print("({})|{} -- {}: {}".format(len(reqs[req_id].vnfs), reqs[req_id].entry_point, N_map_inv[n], len(my_net.paths_links[N_map_inv[n]][reqs[req_id].entry_point])))

    m.addConstrs(
        (
            gp.quicksum(
                q_var[req_id][n][N_map[reqs[req_id].entry_point]][pth_id, len(reqs[req_id].vnfs)]
                for pth_id in range(len(my_net.paths_links[N_map_inv[n]][reqs[req_id].entry_point]))
            ) == v_var[req_id][n, len(reqs[req_id].vnfs)-1]
            for req_id in range(len(reqs))
            for n in Ec_id
        ), name="entry_out"
    )

    m.addConstrs(
        (
            v_var[req_id][n1, i] + v_var[req_id][n2, i+1] - 1 <= gp.quicksum(
                q_var[req_id][n1][n2][pth_id, i+1]
                for pth_id in range(len(my_net.paths_links[N_map_inv[n1]][N_map_inv[n2]]))
            )
            for req_id in range(len(reqs))
            for i in range(len(reqs[req_id].vnfs)-1)
            for n1 in Ec_id
            for n2 in Ec_id
            if n1 != n2
        ), name="chain"
    )

    m.addConstrs(
        (
            gp.quicksum(
                w_var[req_id][N_map[pth[0]]][N_map[pth[1]]][pth[2], R_id[r]] * Rvol[R_id[r]] / len(reqs[req_id].T1)
                for req_id in range(len(reqs))
                if t in reqs[req_id].T1
                for r in R
                for pth in my_net.link_to_path[ll]
            ) + gp.quicksum(
                q_var[req_id][N_map[pth[0]]][N_map[pth[1]]][pth[2], i] * reqs[req_id].vnf_in_rate(i)
                for req_id in range(len(reqs))
                if t in reqs[req_id].T1
                for i in range(len(reqs[req_id].vnfs)+1)
                for pth in my_net.link_to_path[ll]
            ) <= my_net.g[ll[0]][ll[1]]["li"].bw_avail(t)
            for ll in L
            if ll[0] != cloud_node or ll[1] != cloud_node
            for t in T_all
        ), name="bw"
    )

    m.addConstrs(
        (
            gp.quicksum(
                q_var[req_id][n1][n2][pth_id, i] * my_net.get_path_delay(N_map_inv[n1], N_map_inv[n2], pth_id)
                for i in range(len(reqs[req_id].vnfs)+1)
                for n1 in N_map.values()
                for n2 in N_map.values()
                if n1 != n2
                if n1 in Ec_id or n2 in Ec_id
                for pth_id in range(len(my_net.paths_links[N_map_inv[n1]][N_map_inv[n2]]))
            ) <= reqs[req_id].max_delay
            for req_id in range(len(reqs))
        ), name="delay"
    )

    m.setObjective(
        gp.quicksum(
            q_var[req_id][n1][n2][pth_id, i] * len(my_net.paths_links[N_map_inv[n1]][N_map_inv[n2]][pth_id]) * reqs[req_id].vnf_in_rate(i)
            for req_id in range(len(reqs))
            for i in range(len(reqs[req_id].vnfs) + 1)
            for n1 in N_map.values()
            for n2 in N_map.values()
            if n1 != n2
            if n1 in Ec_id or n2 in Ec_id
            for pth_id in range(len(my_net.paths_links[N_map_inv[n1]][N_map_inv[n2]]))
        ), GRB.MINIMIZE
    )

    return m, v_var, q_var, w_var, r_var, T_all, R_id, E_id, Ec_id, N_map, N_map_inv, cloud_node

    # m.setParam("LogToConsole", False)
    # m.setParam("Threads", 6)
    # # m.setParam("TIME_LIMIT", 500)
    # m.optimize()
    # # m.write("out.lp")
    #
    # if m.status == GRB.INFEASIBLE:
    #     # m.computeIIS()
    #     # m.write("s_model.ilp")
    #     return False
    #
    # return True
