import gurobipy as gp
from gurobipy import GRB
from constants import Const
from itertools import chain
from sfc import LayerDownload
from opt_ilp import get_ilp
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


def solve_single_relax(my_net, R, Rvol, req):
    reqs = [req]
    m, v_var, q_var, w_var, r_var, T_all, R_id, E_id, Ec_id, N_map, N_map_inv, cloud_node = get_ilp(reqs, my_net, R, Rvol)

    m.update()
    for v in m.getVars():
        v.setAttr('vtype', 'C')

    m.setParam("LogToConsole", False)
    m.setParam("Threads" , 6)
    # m.setParam("TIME_LIMIT", 500)
    # m.optimize()
    # m.write("out.lp")

    # if m.status == GRB.INFEASIBLE:
    #     m.computeIIS()
    #     m.write("s_model.ilp")
    #     return False, None
    # else:
    #     print(m.objVal)

    Gamma = 2
    gamma = Gamma
    tol_val = 1e-4
    loc_of = dict()
    routing_paths = dict()
    dl_paths = dict()
    total_dl_vol = dict()
    downloads = dict()
    i = 0
    while i < len(req.vnfs):
        # remove edge servers with insufficient resources
        for e in E_id:
            if my_net.g.nodes[N_map_inv[e]]["nd"].cpu_min_avail(req.T2) < req.cpu_req(i):
                v_var[0][e, i].ub = 0
            if my_net.g.nodes[N_map_inv[e]]["nd"].ram_min_avail(req.T2) < req.ram_req(i):
                v_var[0][e, i].ub = 0
            Rd_ei, d_ei = my_net.get_need_storage_layers(N_map_inv[e], req, i, req.tau1)
            if my_net.g.nodes[N_map_inv[e]]["nd"].disk_min_avail_no_cache(req.T2) < d_ei:
                v_var[0][e, i].ub = 0
        # remove paths with insufficient bandwidth
        pvn = req.entry_point if i == 0 else loc_of[i-1]
        for cdn in my_net.paths_links[pvn]:
            for pth_id in range(len(my_net.paths_links[pvn][cdn])):
                if my_net.get_path_min_bw(pvn, cdn, pth_id, req.T2) < req.vnf_in_rate(i):
                    q_var[0][N_map[pvn]][N_map[cdn]][pth_id, i].ub = 0
        # solve to obtain layer download vals
        m.optimize()
        if m.status == GRB.INFEASIBLE:
            if i == 0 or gamma < Gamma:
                return False, None
            elif gamma == Gamma:
                gamma = max(gamma-Gamma, gamma-i)
                i_back = max(0, i-Gamma)
                for ii in range(i_back, i+ 1):
                    total_dl_vol[ii] = 0
                    ## undo vnf placement
                    v_var[0][N_map[loc_of[ii]], ii].lb = 0.0
                    my_net.g.nodes[loc_of[ii]]["nd"].unembed(req, i)
                    ## undo chaining
                    if loc_of[ii-1] != loc_of[ii]:
                        q_var[0][N_map[loc_of[ii-1]]][N_map[loc_of[ii]]][routing_paths[ii], ii].lb = 0.0
                        for ll in my_net.paths_links[loc_of[ii-1]][loc_of[ii]][routing_paths[ii]]:
                            l_obj = my_net.g[ll[0]][ll[1]]["li"]
                            l_obj.unembed(req, i)
                    ## undo download
                    for rr in dl_paths[ii]:
                        w_var[0][N_map[loc_of[ii]]][N_map[cloud_node]][dl_paths[ii][R_id[rr]], rr].lb = 0.0
                    for ld in downloads[ii]:
                        ld.cancel_download()
                i = i_back
                continue
        # get best location
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
        v_var[0][best_loc, i].lb = 1.0
        loc_of[i] = N_map_inv[best_loc]
        my_net.g.nodes[loc_of[i]]["nd"].embed(req, i)
        req.used_servers.add(loc_of[i])
        # fix chain path
        pvn = req.entry_point if i == 0 else loc_of[i - 1]
        cdn = loc_of[i]
        if pvn != cdn:
            path_vals = list()
            for pth_id in range(len(my_net.paths_links[pvn][cdn])):
                a = q_var[0][N_map[pvn]][N_map[cdn]][pth_id, i].x
                path_vals.append(a)
            routing_paths[i] = path_vals.index(max(path_vals))
            for ll in my_net.paths_links[pvn][cdn][routing_paths[i]]:
                l_obj = my_net.g[ll[0]][ll[1]]["li"]
                l_obj.embed(req, i)
            q_var[0][N_map[pvn]][N_map[cdn]][routing_paths[i], i].lb = 1.0
        # determine download possibility
        if best_loc in E_id:
            Rd_ei, _ = my_net.get_missing_layers(N_map_inv[best_loc], req, i, req.tau1)
            for rr in Rd_ei:
                if i not in dl_paths:
                    dl_paths[i] = dict()
                pth_pr = []
                pth_ids = range(len(my_net.paths_links[N_map_inv[best_loc]][cloud_node]))
                for pth_id in pth_ids:
                    a = w_var[0][best_loc][N_map[cloud_node]][pth_id, rr].x
                    pth_pr.append(a)
                dl_paths[i][rr] = np.random.choice(a=pth_ids, p=pth_pr)
                w_var[0][best_loc][N_map[cloud_node]][dl_paths[i][R_id[rr]], rr].lb = 1.0
        m.optimize()
        if m.status == GRB.INFEASIBLE:
            if i == 0 or gamma < Gamma:
                return False, None
            elif gamma == Gamma:
                gamma = max(gamma-Gamma, gamma-i)
                i_back = max(0, i-Gamma)
                for ii in range(i_back, i+ 1):
                    ## undo vnf placement
                    v_var[0][N_map[loc_of[ii]], ii].lb = 0.0
                    my_net.g.nodes[loc_of[ii]]["nd"].unembed(req, i)
                    ## undo chaining
                    if loc_of[ii-1] != loc_of[ii]:
                        q_var[0][N_map[loc_of[ii-1]]][N_map[loc_of[ii]]][routing_paths[ii], ii].lb = 0.0
                        for ll in my_net.paths_links[loc_of[ii-1]][loc_of[ii]][routing_paths[ii]]:
                            l_obj = my_net.g[ll[0]][ll[1]]["li"]
                            l_obj.unembed(req, i)
                    ## undo download
                    for rr in dl_paths[ii]:
                        w_var[0][N_map[loc_of[ii]]][N_map[cloud_node]][dl_paths[ii][R_id[rr]], rr].lb = 0.0
                    for ld in downloads[ii]:
                        ld.cancel_download()
                i = i_back
                continue
        Rd_ei, _ = my_net.get_missing_layers(loc_of[i], req, i, req.tau1)
        my_net.g.nodes[loc_of[i]]["nd"].add_layer(Rd_ei, req)
        downloads[i] = set()
        total_dl_vol[i] = 0
        for rr in dl_paths[i]:
            total_dl_vol[i] = total_dl_vol[i] + Rvol[rr]
            layer_download = LayerDownload()
            downloads[i].add(layer_download)
            pp = dl_paths[i][rr]
            for tt in req.T1:
                for ll in my_net.paths_links[loc_of[i]][cloud_node][pp]:
                    l_obj = my_net.g[ll[0]][ll[1]]["li"]
                    layer_download.add_data(tt, l_obj, Rvol[rr] / len(req.T1))
        # go to next vnf
        i = i + 1
        gamma = min(Gamma, gamma+1)

    path_vals = list()
    pvn = loc_of[len(req.vnfs)-1]
    cdn = req.entry_point
    for pth_id in range(len(my_net.paths_links[pvn][cdn])):
        a = q_var[0][N_map[pvn]][N_map[cdn]][pth_id, i].x
        path_vals.append(a)
    routing_paths[i] = path_vals.index(max(path_vals))
    for ll in my_net.paths_links[pvn][cdn][routing_paths[i]]:
        l_obj = my_net.g[ll[0]][ll[1]]["li"]
        l_obj.embed(req, i)
    q_var[0][N_map[pvn]][N_map[cdn]][routing_paths[i], i].lb = 1.0

    m.optimize()
    if m.status == GRB.INFEASIBLE:
        my_net.evict_sfc(req)
        for ii in downloads:
            for ld in downloads[ii]:
                ld.cancel_download()
        return False, None

    return True, sum(total_dl_vol.values())
