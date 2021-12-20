import gurobipy as gp
from gurobipy import GRB
from constants import Const
from itertools import chain
from sfc import LayerDownload
from opt_ilp import get_ilp
import numpy as np


def solve_single_relax(my_net, R, Rvol, req, Gamma, bw_scaler):
    reqs = [req]
    m, v_var, q_var, w_var, r_var, T_all, R_id, E_id, Ec_id, N_map, N_map_inv, cloud_node = get_ilp(reqs, my_net, R, Rvol)

    m.update()
    for v in m.getVars():
        v.setAttr('vtype', 'C')
        v.lb = 0.0
        v.ub = 1.0

    m.setParam("LogToConsole", False)
    m.setParam("Threads" , 6)
    # m.setParam("TIME_LIMIT", 500)
    # m.optimize()
    # m.write("out.lp")

    if m.status == GRB.INFEASIBLE:
        # m.computeIIS()
        # m.write("s_model.ilp")
        return False, 0, 0
    # else:
    #     print(m.objVal)

    Lw, L_iii = my_net.get_link_sets()
    gamma = Gamma
    tol_val = 1e-4
    loc_of = dict()
    routing_paths = dict()
    dl_paths = dict()
    total_dl_vol = dict()
    downloads = dict()
    v_eliminations = dict()
    q_eliminations = dict()
    i = 0
    while i < len(req.vnfs):
        v_eliminations[i] = set()
        q_eliminations[i] = set()
        # remove edge servers with insufficient resources
        for e in E_id:
            if my_net.g.nodes[N_map_inv[e]]["nd"].cpu_min_avail(req.T2) < req.cpu_req(i):
                v_var[0][e, i].ub = 0
                v_eliminations[i].add(e)
            if my_net.g.nodes[N_map_inv[e]]["nd"].ram_min_avail(req.T2) < req.ram_req(i):
                v_var[0][e, i].ub = 0
                v_eliminations[i].add(e)
            Rd_ei, d_ei = my_net.get_need_storage_layers(N_map_inv[e], req, i, req.tau1)
            if my_net.g.nodes[N_map_inv[e]]["nd"].disk_min_avail_no_cache(req.T2) < d_ei:
                v_var[0][e, i].ub = 0
                v_eliminations[i].add(e)
        # remove paths with insufficient bandwidth
        pvn = req.entry_point if i == 0 else loc_of[i-1]
        for cdn in my_net.paths_links[pvn]:
            for pth_id in range(len(my_net.paths_links[pvn][cdn])):
                if my_net.get_path_min_bw(pvn, cdn, pth_id, req.T2) < req.vnf_in_rate(i):
                    q_var[0][N_map[pvn]][N_map[cdn]][pth_id, i].ub = 0
                    q_eliminations[i].add((N_map[pvn], N_map[cdn], pth_id))
        # scale links
        link_time = dict()
        if bw_scaler < 1.0:
            for tt in req.T1:
                for ll in Lw:
                    if ll[0] != cloud_node or ll[1] != cloud_node:
                        if ll[0][0] != 'b' and ll[1][0] != 'b':
                            cname = "bw[('{}', '{}'),{}]".format(ll[0], ll[1], tt)
                            cc = m.getConstrByName(cname)
                            if cc is not None:
                                link_time[(ll, tt)] = cc.getAttr(GRB.Attr.RHS)
                                cc.setAttr(GRB.Attr.RHS, bw_scaler * cc.getAttr(GRB.Attr.RHS))
        # solve to obtain loc
        m.optimize()
        if bw_scaler < 1.0:
            for ll, tt in link_time:
                cname = "bw[('{}', '{}'),{}]".format(ll[0], ll[1], tt)
                cc = m.getConstrByName(cname)
                cc.setAttr(GRB.Attr.RHS, link_time[(ll, tt)])
        if m.status == GRB.INFEASIBLE:
            # m.computeIIS()
            # m.write("s_model.ilp")
            if i == 0 or gamma < Gamma or Gamma == 0:
                print("one failed after elimination!")
                for ii in range(len(req.vnfs)):
                    if ii in loc_of:
                        my_net.g.nodes[loc_of[ii]]["nd"].unembed(req, ii)
                my_net.evict_sfc(req)
                for ii in downloads:
                    for ld in downloads[ii]:
                        ld.cancel_download()
                return False, 0, 0
            elif gamma == Gamma:
                print("Doing a backtack!")
                gamma = max(gamma-Gamma-1, gamma-i-1)
                i_back = max(0, i-Gamma)
                for ii in range(i_back, i+1):
                    total_dl_vol[ii] = 0
                    ##
                    for ee in v_eliminations[ii]:
                        v_var[0][ee, ii].ub = 1
                        v_var[0][ee, ii].lb = 0
                    for n1, n2, pp in q_eliminations[ii]:
                        q_var[0][n1][n2][pp, ii].ub = 1
                        q_var[0][n1][n2][pp, ii].lb = 0
                    ## undo vnf placement
                    if ii in loc_of:
                        v_var[0][N_map[loc_of[ii]], ii].lb = 0.0
                        my_net.g.nodes[loc_of[ii]]["nd"].unembed(req, ii)
                        ## undo chaining
                        pvn = req.entry_point if ii == 0 else loc_of[ii - 1]
                        cdn = loc_of[ii]
                        if pvn != cdn:
                            q_var[0][N_map[pvn]][N_map[cdn]][routing_paths[ii], ii].lb = 0.0
                            for ll in my_net.paths_links[pvn][cdn][routing_paths[ii]]:
                                l_obj = my_net.g[ll[0]][ll[1]]["li"]
                                l_obj.unembed(req, ii)
                        ## undo download
                        if ii in dl_paths:
                            for rr in dl_paths[ii]:
                                w_var[0][N_map[loc_of[ii]]][N_map[cloud_node]][dl_paths[ii][R_id[rr]], rr].lb = 0.0
                            del dl_paths[ii]
                        if ii in downloads:
                            for ld in downloads[ii]:
                                ld.cancel_download()
                            del downloads[ii]
                i = i_back
                v_var[0][N_map[loc_of[i]], i].ub = 0.0
                continue
        # get best location
        loc_pr = list()
        for n in Ec_id:
            loc_pr.append(v_var[0][n, i].x)
        best_loc = Ec_id[loc_pr.index(max(loc_pr))]
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
        # solve to obtain layer download vals
        # for cc in m.getConstrs():
        #     if cc.ConstrName[0:2] == "bw":
        #         print(cc.ConstrName)
        rounding_failed = False
        Rd_ei, _ = my_net.get_missing_layers(N_map_inv[best_loc], req, i, req.tau1)
        for rr in Rd_ei:
            if i not in dl_paths:
                dl_paths[i] = dict()
            pth_pr = []
            pth_ids = range(len(my_net.paths_links[N_map_inv[best_loc]][cloud_node]))
            for pth_id in pth_ids:
                a = w_var[0][best_loc][N_map[cloud_node]][pth_id, rr].x
                pth_pr.append(a)
            if sum(pth_pr) != 0.0:
                if sum(pth_pr) < 1.0:
                    pth_pr = [pr/sum(pth_pr) for pr in pth_pr]
                dl_paths[i][rr] = np.random.choice(a=pth_ids, p=pth_pr)
                w_var[0][best_loc][N_map[cloud_node]][dl_paths[i][R_id[rr]], rr].lb = 1.0
            else:
                print("one failed, no candidate path!")
                rounding_failed = True
                break
        if not rounding_failed:
            m.optimize()
        if rounding_failed or m.status == GRB.INFEASIBLE:
            # m.computeIIS()
            # m.write("s_model.ilp")
            if i == 0 or gamma < Gamma or Gamma == 0:
                if not rounding_failed:
                    print("one failed after rounding!")
                for ii in range(len(req.vnfs)):
                    if ii in loc_of:
                        my_net.g.nodes[loc_of[ii]]["nd"].unembed(req, ii)
                my_net.evict_sfc(req)
                for ii in downloads:
                    for ld in downloads[ii]:
                        ld.cancel_download()
                return False, None, 0
            elif gamma == Gamma:
                print("Doing a backtack!")
                gamma = max(gamma - Gamma - 1, gamma - i - 1)
                i_back = max(0, i - Gamma)
                for ii in range(i_back, i + 1):
                    total_dl_vol[ii] = 0
                    ##
                    for ee in v_eliminations[ii]:
                        v_var[0][ee, ii].ub = 1
                        v_var[0][ee, ii].lb = 0
                    for n1, n2, pp in q_eliminations[ii]:
                        q_var[0][n1][n2][pp, ii].ub = 1
                        q_var[0][n1][n2][pp, ii].lb = 0
                    ## undo vnf placement
                    if ii in loc_of:
                        v_var[0][N_map[loc_of[ii]], ii].lb = 0.0
                        my_net.g.nodes[loc_of[ii]]["nd"].unembed(req, ii)
                        ## undo chaining
                        pvn = req.entry_point if ii == 0 else loc_of[ii - 1]
                        cdn = loc_of[ii]
                        if pvn != cdn:
                            q_var[0][N_map[pvn]][N_map[cdn]][routing_paths[ii], ii].lb = 0.0
                            for ll in my_net.paths_links[pvn][cdn][routing_paths[ii]]:
                                l_obj = my_net.g[ll[0]][ll[1]]["li"]
                                l_obj.unembed(req, ii)
                        ## undo download
                        if ii in dl_paths:
                            for rr in dl_paths[ii]:
                                w_var[0][N_map[loc_of[ii]]][N_map[cloud_node]][dl_paths[ii][R_id[rr]], rr].lb = 0.0
                            del dl_paths[ii]
                        if ii in downloads:
                            for ld in downloads[ii]:
                                ld.cancel_download()
                            del downloads[ii]
                i = i_back
                v_var[0][N_map[loc_of[i]], i].ub = 0.0
                continue
        ##
        my_net.g.nodes[loc_of[i]]["nd"].add_layer(req.vnfs[i].layers, req)
        downloads[i] = set()
        total_dl_vol[i] = 0
        if i in dl_paths:
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
        print("one failed at last!")
        for ii in range(len(req.vnfs)):
            if ii in loc_of:
                my_net.g.nodes[loc_of[ii]]["nd"].unembed(req, ii)
        my_net.evict_sfc(req)
        for ii in downloads:
            for ld in downloads[ii]:
                ld.cancel_download()
        return False, 0, 0
    else:
        print("one success!")
        for ii in range(len(req.vnfs)):
            if ii in loc_of:
                my_net.g.nodes[loc_of[ii]]["nd"].finalize_layer()
        return True, sum(total_dl_vol.values()), m.objVal
