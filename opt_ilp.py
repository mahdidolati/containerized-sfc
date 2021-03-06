# RCCO, (C) Mahdi Dolati. License: AGPLv3
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from test import TestResult


class IlpModel:
    def __init__(self):
        self.m = None
        self.v_var = None
        self.q_var = None
        self.w_var = None
        self.r_var = None
        self.T_all = None
        self.E_id = None
        self.Ec_id = None
        self.N_map = None
        self.N_map_inv = None
        self.cloud_node = None


def get_T(reqs):
    t1 = np.infty
    t2 = 0
    for r in reqs:
        t1 = min(r.arrival_time, t1)
        t2 = max(r.tau2, t2)
    return range(t1, t2+1)


def solve_batch_opt(reqs, my_net, R, Rvol):
    req_len = len(reqs)
    a_reqs = list()
    tr = TestResult()
    ilp_model = None
    for req_id in range(req_len):
        ilp_model2 = get_ilp(a_reqs + [reqs[req_id]], my_net, R, Rvol, ilp_model)
        ilp_model2.m.setParam("LogToConsole", False)
        ilp_model2.m.setParam("Threads", 6)
        # ilp_model.m.setParam("TIME_LIMIT", 100)
        ilp_model2.m.optimize()
        # m.write("out.lp")

        if ilp_model2.m.status == GRB.INFEASIBLE or ilp_model2.m.status == GRB.INF_OR_UNBD or ilp_model2.m.getAttr("SolCount") <= 0:
            # m.computeIIS()
            # m.write("s_model.ilp")
            # return False, 1, 0
            print("rejected one!")
            tr.res_groups[tr.SF] = tr.res_groups[tr.SF] + 1
        else:
            ilp_model = ilp_model2
            a_reqs.append(reqs[req_id])
            for vnf_id in range(len(reqs[req_id].vnfs) + 1):
                tr.revenue = tr.revenue + reqs[req_id].vnf_in_rate(vnf_id)
            print(ilp_model.m.objVal)
            tr.avg_admit = 1.0 * (len(a_reqs)) / req_len
            tr.chain_bw = ilp_model.m.objVal
            tr.res_groups[tr.SU] = tr.res_groups[tr.SU] + 1

    dl_layer = dict()
    if ilp_model is not None:
        for req_id in range(len(a_reqs)):
            for e in ilp_model.E_id:
                for i in range(len(a_reqs[req_id].vnfs)):
                    if ilp_model.v_var[req_id][e, i].x > 1 - 0.0001:
                        if e not in dl_layer:
                            dl_layer[e] = set()
                        for l in a_reqs[req_id].vnfs[i].layers:
                            if l not in dl_layer[e]:
                                tr.avg_dl = tr.avg_dl + a_reqs[req_id].vnfs[i].layers[l]
                                dl_layer[e].add(l)

    return tr


def solve_batch_opt2(reqs, my_net, R, Rvol):
    req_len = len(reqs)
    tr = TestResult()
    ilp_model = get_ilp(reqs, my_net, R, Rvol, None)
    ilp_model.m.setParam("LogToConsole", False)
    ilp_model.m.setParam("Threads", 6)
    # ilp_model2.m.setParam("TIME_LIMIT", 100)
    ilp_model.m.optimize()
    # m.write("out.lp")

    if ilp_model.m.status == GRB.INFEASIBLE or ilp_model.m.status == GRB.INF_OR_UNBD or ilp_model.m.getAttr("SolCount") <= 0:
        # m.computeIIS()
        # m.write("s_model.ilp")
        # return False, 1, 0
        print("rejected one!")
        tr.res_groups[tr.SF] = len(reqs)
    else:
        # a_reqs.append(reqs[req_id])
        for req_id in range(req_len):
            for vnf_id in range(len(reqs[req_id].vnfs) + 1):
                tr.revenue = tr.revenue + reqs[req_id].vnf_in_rate(vnf_id)
        print(ilp_model.m.objVal)
        tr.avg_admit = 1.0 * (len(reqs)) / req_len
        tr.chain_bw = ilp_model.m.objVal
        tr.res_groups[tr.SU] = len(reqs)
        dl_layer = dict()
        for req_id in range(len(reqs)):
            for e in ilp_model.E_id:
                for i in range(len(reqs[req_id].vnfs)):
                    if ilp_model.v_var[req_id][e, i].x > 1 - 0.0001:
                        if e not in dl_layer:
                            dl_layer[e] = set()
                        for l in reqs[req_id].vnfs[i].layers:
                            if l not in dl_layer[e]:
                                tr.avg_dl = tr.avg_dl + reqs[req_id].vnfs[i].layers[l]
                                dl_layer[e].add(l)
    return tr


def get_ilp(reqs, my_net, R, Rvol, ilp_model=None):
    # t1 = process_time()
    T_all = get_T(reqs)
    B = my_net.get_all_base_stations()
    E = my_net.get_all_edge_nodes()
    cloud_node = "c"
    Ec = E + [cloud_node]
    N = Ec + B

    L, L_iii = my_net.get_link_sets()

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
                    w_var[req_id][N_map[n1]][N_map[n2]] = m.addVars(len(my_net.paths_links[n1][n2]), R,
                                                      vtype=GRB.BINARY, name="w,{},{},{},".format(req_id,n1,n2))
    r_var = m.addVars(E_id, R, T_all, vtype=GRB.BINARY, name="r")

    if ilp_model is not None:
        for req_id in range(len(reqs)-1):
            for n in Ec_id:
                for i in range(len(reqs[req_id].vnfs)):
                    if ilp_model.v_var[req_id][n, i].x > 1.0 - 0.0001:
                        v_var[req_id][n, i].lb = 1.0
                        break
            for n1 in my_net.paths_links:
                for n2 in my_net.paths_links[n1]:
                    for j in range(len(my_net.paths_links[n1][n2])):
                        for i in range(len(reqs[req_id].vnfs)+1):
                            if ilp_model.q_var[req_id][N_map[n1]][N_map[n2]][j, i].x > 1.0 - 0.0001:
                                q_var[req_id][N_map[n1]][N_map[n2]][j, i].lb = 1.0


    # t2 = process_time()
    # print("Vars done {}".format(t2-t1))
    # t1 = t2

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

    # t2 = process_time()
    # print("placement_all done {}".format(t2 - t1))
    # t1 = t2

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

    # t2 = process_time()
    # print("cpu_limit done {}".format(t2 - t1))
    # t1 = t2

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

    # t2 = process_time()
    # print("ram_limit done {}".format(t2 - t1))
    # t1 = t2
    # print("rid", R_id)
    # for req_id in range(len(reqs)):
    #     for i in range(len(reqs[req_id].vnfs)):
    #         print(i, reqs[req_id].vnfs[i].layers)

    m.addConstrs(
        (
            v_var[req_id][e, i] <= r_var[e, r, t]
            for e in E_id
            for req_id in range(len(reqs))
            for i in range(len(reqs[req_id].vnfs))
            for r in reqs[req_id].vnfs[i].layers
            for t in reqs[req_id].T2
        ), name="layer_prereq"
    )

    # t2 = process_time()
    # print("layer_prereq done {}".format(t2 - t1))
    # t1 = t2

    if 0 in T_all:
        m.addConstrs(
            (
                r_var[e, r, 0] == 0
                for e in E_id
                for r in R
            ), name="layer_0"
        )
        # t2 = process_time()
        # print("layer_0 done {}".format(t2 - t1))
        # t1 = t2

    # t1 = process_time()
    for e in E_id:
        for r in R:
            for t in T_all:
                if t == 0:
                    continue
                if my_net.g.nodes[N_map_inv[e]]["nd"].layer_avail(r, t):
                    if my_net.g.nodes[N_map_inv[e]]["nd"].layer_inuse(r):
                        r_var[e, r, t].lb = 1.0
                    else:
                        pass # may assume or not with no problem
                else:
                    if t-1 not in T_all:
                        m.addConstr(
                            r_var[e, r, t] <= gp.quicksum(
                                w_var[req_id][e][N_map[cloud_node]][pth_id, r]
                                for req_id in range(len(reqs))
                                if r in reqs[req_id].layers
                                if t in reqs[req_id].T2
                                for pth_id in range(len(my_net.paths_links[N_map_inv[e]][cloud_node]))
                            ), name="layer_download,{},{},{}".format(e, r, t)
                        )
                    else:
                        m.addConstr(
                            r_var[e, r, t] <= r_var[e, r, t - 1] + gp.quicksum(
                                w_var[req_id][e][N_map[cloud_node]][pth_id, r]
                                for req_id in range(len(reqs))
                                if r in reqs[req_id].layers
                                if t in reqs[req_id].T2
                                for pth_id in range(len(my_net.paths_links[N_map_inv[e]][cloud_node]))
                            ), name="layer_download,{},{},{}".format(e, r, t)
                        )
    # t2 = process_time()
    # print("layer_download done {}".format(t2 - t1))
    # t1 = t2

    m.addConstrs(
        (
            gp.quicksum(
                w_var[req_id][e][N_map[cloud_node]][pth_id, r]
                for pth_id in range(len(my_net.paths_links[N_map_inv[e]][cloud_node]))
            ) <= 1
            for req_id in range(len(reqs))
            for e in E_id
            for r in R
        ), name="download_path_1"
    )

    # t2 = process_time()
    # print("download_path_1 done {}".format(t2 - t1))
    # t1 = t2

    m.addConstrs(
        (
            gp.quicksum(
                r_var[e, r, t] * Rvol[r]
                for r in R
            ) <= my_net.g.nodes[N_map_inv[e]]["nd"].disk
            for e in E_id
            for t in T_all
        ), name="disk"
    )

    # t2 = process_time()
    # print("disk done {}".format(t2 - t1))
    # t1 = t2

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

    # t2 = process_time()
    # print("entry_in done {}".format(t2 - t1))
    # t1 = t2

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

    # t2 = process_time()
    # print("entry_out done {}".format(t2 - t1))
    # t1 = t2

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

    # t2 = process_time()
    # print("chain done {}".format(t2 - t1))
    # t1 = t2
    #
    # print("L: {}, T: {}, Rq: {}, R: {}".format(len(L), len(T_all), len(reqs), len(R)))
    # print("L: {}".format(L))
    # print("R_id: {}".format(R_id))
    # print("R_vol: {}".format(Rvol))

    m.addConstrs(
        (
            gp.quicksum(
                w_var[req_id][N_map[pth[0]]][N_map[pth[1]]][pth[2], r] * Rvol[r] / len(reqs[req_id].T1)
                for req_id in range(len(reqs))
                if t in reqs[req_id].T1
                for r in reqs[req_id].layers
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
            if ll in my_net.link_to_path
            for t in T_all
        ), name="bw"
    )

    # t2 = process_time()
    # print("bw done {}".format(t2 - t1))
    # t1 = t2

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

    # t2 = process_time()
    # print("delay done {}".format(t2 - t1))
    # t1 = t2

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

    # t2 = process_time()
    # print("setObjective done {}".format(t2 - t1))
    # t1 = t2
    ilp_model = IlpModel()
    ilp_model.m = m
    ilp_model.v_var = v_var
    ilp_model.q_var = q_var
    ilp_model.w_var = w_var
    ilp_model.r_var = r_var
    ilp_model.T_all = T_all
    ilp_model.E_id = E_id
    ilp_model.Ec_id = Ec_id
    ilp_model.N_map = N_map
    ilp_model.N_map_inv = N_map_inv
    ilp_model.cloud_node = cloud_node

    return ilp_model

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
