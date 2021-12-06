import gurobipy as gp
from gurobipy import GRB
from constants import Const
from itertools import chain
from sfc import LayerDownload
from opt_ilp import get_ilp


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


def solve_single_relax(my_net, R, Rvol, req):
    reqs = [req]
    m, v_var, q_var, w_var, r_var, T_all, R_id, E_id, Ec_id, N_map, N_map_inv, cloud_node = get_ilp(reqs, my_net, R, Rvol)

    m.update()
    for v in m.getVars():
        v.setAttr('vtype', 'C')

    m.setParam("LogToConsole", False)
    m.setParam("Threads" , 6)
    # m.setParam("TIME_LIMIT", 500)
    m.optimize()
    # m.write("out.lp")

    if m.status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write("s_model.ilp")
        return False, None
    else:
        print(m.objVal)

    tol_val = 1e-4
    loc_of = dict()
    dl_path_of = dict()
    routing_paths = dict()
    for i in range(len(req.vnfs)):
        best_loc = None
        for ttt in range(3):
            v_max = 0
            best_loc = None
            for n in Ec_id:
                a = v_var[0][n, i].x
                if a > tol_val:
                    if best_loc is None or v_max < a:
                        best_loc = n
                        v_max = a
                if v_max >= 1.0 - tol_val:
                    break
            if best_loc is None:
                return False, None
            v_var[0][best_loc, i].lb = 1.0
            m.optimize()
            if m.status != GRB.INFEASIBLE:
                break
            else:
                v_var[0][best_loc, i].lb = 0.0
                v_var[0][best_loc, i].ub = 0.0
                m.optimize()
        if m.status == GRB.INFEASIBLE:
            return False, None
        loc_of[i] = N_map_inv[best_loc]

        if best_loc in E_id:
            Rd_ei, _ = my_net.get_missing_layers(N_map_inv[best_loc], req, i, req.tau1)
            for rr in Rd_ei:
                y_max = 0
                best_path = None
                for pth_id in range(len(my_net.paths_links[N_map_inv[best_loc]][cloud_node])):
                    a = w_var[0][best_loc][N_map[cloud_node]][pth_id, rr].x
                    if a > tol_val:
                        if best_path is None or y_max < a:
                            best_path = pth_id
                            y_max = a
                    if y_max >= 1.0 - tol_val:
                        break
                if best_path is not None:
                    dl_path_of[(loc_of[i], R_id[rr])] = best_path
                    w_var[0][best_loc][N_map[cloud_node]][best_path, rr].lb = 1.0
                else:
                    return False, None

        if i == 0:
            q_max = 0
            best_path = None
            for pth_id in range(len(my_net.paths_links[req.entry_point][loc_of[i]])):
                a = q_var[0][N_map[req.entry_point]][best_loc][pth_id, i].x
                if a > tol_val:
                    if best_path is None or q_max < a:
                        best_path = pth_id
                        q_max = a
                if q_max >= 1.0 - tol_val:
                    break
            if best_path is not None:
                routing_paths[i] = my_net.paths_links[req.entry_point][loc_of[i]][best_path]
                q_var[0][N_map[req.entry_point]][best_loc][best_path, i].lb = 1.0
            else:
                return False, None
        elif loc_of[i-1] != loc_of[i]:
            q_max = 0
            best_path = None
            for pth_id in range(len(my_net.paths_links[loc_of[i-1]][loc_of[i]])):
                a = q_var[0][N_map[loc_of[i-1]]][best_loc][pth_id, i].x
                if a > tol_val:
                    if best_path is None or q_max < a:
                        best_path = pth_id
                        q_max = a
                if q_max >= 1.0 - tol_val:
                    break
            if best_path is not None:
                routing_paths[i] = my_net.paths_links[loc_of[i-1]][loc_of[i]][best_path]
                q_var[0][N_map[loc_of[i-1]]][best_loc][best_path, i].lb = 1.0
            else:
                return False, None

        m.optimize()
        if m.status == GRB.INFEASIBLE:
            return False, None

    q_max = 0
    best_path = None
    for pth_id in range(len(my_net.paths_links[loc_of[len(req.vnfs)-1]][req.entry_point])):
        a = q_var[0][N_map[loc_of[len(req.vnfs)-1]]][N_map[req.entry_point]][pth_id, len(req.vnfs)].x
        if a > tol_val:
            if best_path is None or q_max < a:
                best_path = pth_id
                q_max = a
        if q_max >= 1.0 - tol_val:
            break
    if best_path is not None:
        routing_paths[len(req.vnfs)] = my_net.paths_links[loc_of[len(req.vnfs)-1]][req.entry_point][best_path]
        q_var[0][N_map[loc_of[len(req.vnfs) - 1]]][N_map[req.entry_point]][best_path, len(req.vnfs)].lb = 1.0
    else:
        return False, None

    m.optimize()
    if m.status == GRB.INFEASIBLE:
        return False, None

    for i in range(len(req.vnfs)):
        my_net.g.nodes[loc_of[i]]["nd"].embed(req, i)
        req.used_servers.add(loc_of[i])
        if loc_of[i][0] == "e":
            Rd_ei, _ = my_net.get_missing_layers(loc_of[i], req, i, req.tau1)
            my_net.g.nodes[loc_of[i]]["nd"].add_layer(Rd_ei, req)

    total_dl_vol = 0
    downloads = []
    for ee, rr in dl_path_of:
        total_dl_vol = total_dl_vol + Rvol[rr]
        layer_download = LayerDownload()
        downloads.append(layer_download)
        pp = dl_path_of[(ee, rr)]
        for tt in req.T1:
            for ll in my_net.paths_links[ee][cloud_node][pp]:
                l_obj = my_net.g[ll[0]][ll[1]]["li"]
                layer_download.add_data(tt, l_obj, Rvol[rr] / len(req.T1))

    for i in range(len(req.vnfs)+1):
        if i in routing_paths:
            for ll in routing_paths[i]:
                l_obj = my_net.g[ll[0]][ll[1]]["li"]
                l_obj.embed(req, i)

    return True, total_dl_vol
