import gurobipy as gp
from gurobipy import GRB
from constants import Const
from itertools import chain
from sfc import LayerDownload
from opt_ilp import get_ilp
import numpy as np


def solve_single(my_net, R, Rvol, req):
    reqs = [req]
    m, v_var, q_var, w_var, r_var, T_all, R_id, E_id, Ec_id, N_map, N_map_inv, cloud_node = get_ilp(reqs, my_net, R, Rvol)

    m.setParam("LogToConsole", False)
    m.setParam("Threads" , 6)
    m.setParam("TIME_LIMIT", 30)
    m.optimize()
    # m.write("out.lp")

    if m.status == GRB.INFEASIBLE:
        print("one failed!")
        # m.computeIIS()
        # m.write("s_model.ilp")
        return False, 0, 0

    loc_of = dict()
    routing_paths = dict()
    dl_paths = dict()
    total_dl_vol = dict()
    downloads = dict()
    for i in range(len(req.vnfs)):
        # vnf
        loc_pr = list()
        for n in Ec_id:
            loc_pr.append(v_var[0][n, i].x)
        best_loc = Ec_id[loc_pr.index(max(loc_pr))]
        loc_of[i] = N_map_inv[best_loc]
        my_net.g.nodes[loc_of[i]]["nd"].embed(req, i)
        req.used_servers.add(loc_of[i])
        # chain
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
        # dl
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
                dl_paths[i][rr] = pth_ids[pth_pr.index(max(pth_pr))]
        #
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

    print("one success!")
    return True, sum(total_dl_vol.values()), m.objVal
