import gurobipy as gp
from gurobipy import GRB
from constants import Const
from itertools import chain
from sfc import LayerDownload
from opt_ilp import get_ilp
import numpy as np
from time import process_time
from test import TestResult


class RelaxSingle:
    FAILED = "FAILED"
    SOL_NO_SCALE = "SOL_NO_SCALE"

    def __init__(self, my_net, R, Rvol, Gamma, bw_scaler):
        self.my_net = my_net
        self.R = R
        self.Rvol = Rvol
        self.Gamma = Gamma
        self.bw_scaler = bw_scaler
        self.ilp_model = None
        self.Lw, self.L_iii = self.my_net.get_link_sets()
        self.tol_val = 1e-4
        self.loc_of = dict()
        self.routing_paths = dict()
        self.dl_paths = dict()
        self.total_dl_vol = dict()
        self.downloads = dict()
        self.v_eliminations = dict()
        self.q_eliminations = dict()

    def eliminate(self, req, i):
        self.v_eliminations[i] = set()
        self.q_eliminations[i] = set()
        # remove edge servers with insufficient resources
        for e in self.ilp_model.E_id:
            if self.my_net.g.nodes[self.ilp_model.N_map_inv[e]]["nd"].cpu_min_avail(req.T2) < req.cpu_req(i):
                self.ilp_model.v_var[0][e, i].ub = 0
                self.v_eliminations[i].add(e)
            if self.my_net.g.nodes[self.ilp_model.N_map_inv[e]]["nd"].ram_min_avail(req.T2) < req.ram_req(i):
                self.ilp_model.v_var[0][e, i].ub = 0
                self.v_eliminations[i].add(e)
            Rd_ei, d_ei = self.my_net.get_need_storage_layers(self.ilp_model.N_map_inv[e], req, i, req.tau1)
            if self.my_net.g.nodes[self.ilp_model.N_map_inv[e]]["nd"].disk_min_avail_no_cache(req.T2) < d_ei:
                self.ilp_model.v_var[0][e, i].ub = 0
                self.v_eliminations[i].add(e)
        # remove paths with insufficient bandwidth
        pvn = req.entry_point if i == 0 else self.loc_of[i - 1]
        for cdn in self.my_net.paths_links[pvn]:
            for pth_id in range(len(self.my_net.paths_links[pvn][cdn])):
                if self.my_net.get_path_min_bw(pvn, cdn, pth_id, req.T2) < req.vnf_in_rate(i):
                    self.ilp_model.q_var[0][self.ilp_model.N_map[pvn]][self.ilp_model.N_map[cdn]][pth_id, i].ub = 0
                    self.q_eliminations[i].add((self.ilp_model.N_map[pvn], self.ilp_model.N_map[cdn], pth_id))

    def scale_links(self, req):
        link_time = dict()
        for tt in req.T1:
            for ll in self.Lw:
                if ll[0] != self.ilp_model.cloud_node or ll[1] != self.ilp_model.cloud_node:
                    if ll[0][0] != 'b' and ll[1][0] != 'b':
                        cname = "bw[('{}', '{}'),{}]".format(ll[0], ll[1], tt)
                        cc = self.ilp_model.m.getConstrByName(cname)
                        if cc is not None:
                            link_time[(ll, tt)] = cc.getAttr(GRB.Attr.RHS)
                            cc.setAttr(GRB.Attr.RHS, self.bw_scaler * cc.getAttr(GRB.Attr.RHS))
        return link_time

    def rescale_links(self, req, link_time):
        for ll, tt in link_time:
            cname = "bw[('{}', '{}'),{}]".format(ll[0], ll[1], tt)
            cc = self.ilp_model.m.getConstrByName(cname)
            cc.setAttr(GRB.Attr.RHS, link_time[(ll, tt)])

    def cleanup(self, req):
        for ii in range(len(req.vnfs)):
            if ii in self.loc_of:
                self.my_net.g.nodes[self.loc_of[ii]]["nd"].unembed(req, ii)
        self.my_net.evict_sfc(req)
        for ii in self.downloads:
            for ld in self.downloads[ii]:
                ld.cancel_download()

    def undo_chaining(self, req, i):
        pvn = req.entry_point if i == 0 else self.loc_of[i - 1]
        cdn = self.loc_of[i]
        if pvn != cdn:
            self.ilp_model.q_var[0][self.ilp_model.N_map[pvn]][self.ilp_model.N_map[cdn]][
                self.routing_paths[i], i].lb = 0.0
            for ll in self.my_net.paths_links[pvn][cdn][self.routing_paths[i]]:
                l_obj = self.my_net.g[ll[0]][ll[1]]["li"]
                l_obj.unembed(req, i)

    def undo(self, req, i):
        self.total_dl_vol[i] = 0
        for ee in self.v_eliminations[i]:
            self.ilp_model.v_var[0][ee, i].ub = 1
            self.ilp_model.v_var[0][ee, i].lb = 0
        for n1, n2, pp in self.q_eliminations[i]:
            self.ilp_model.q_var[0][n1][n2][pp, i].ub = 1
            self.ilp_model.q_var[0][n1][n2][pp, i].lb = 0
        ## undo vnf placement
        if i in self.loc_of:
            self.ilp_model.v_var[0][self.ilp_model.N_map[self.loc_of[i]], i].lb = 0.0
            self.my_net.g.nodes[self.loc_of[i]]["nd"].unembed(req, i)
            ## undo chaining
            self.undo_chaining(req, i)
            ## undo download
            if i in self.dl_paths:
                for rr in self.dl_paths[i]:
                    self.ilp_model.w_var[0][self.ilp_model.N_map[self.loc_of[i]]][self.ilp_model.N_map[self.ilp_model.cloud_node]][self.dl_paths[i][self.ilp_model.R_id[rr]], rr].lb = 0.0
                del self.dl_paths[i]
            if i in self.downloads:
                for ld in self.downloads[i]:
                    ld.cancel_download()
                del self.downloads[i]

    def handle_backtrack(self, req, i, first_bt, gamma, scaled):
        if i != len(req.vnfs) and scaled:
            self.undo(req, i)
            return True, i, first_bt, gamma, not scaled

        if i == 0 and first_bt > 0:
            if i in self.loc_of:
                self.undo(req, i)
                self.ilp_model.v_var[0][self.ilp_model.N_map[self.loc_of[i]], i].lb = 0.0
                self.ilp_model.v_var[0][self.ilp_model.N_map[self.loc_of[i]], i].ub = 0.0
                del self.loc_of[i]
                return True, i, first_bt-1, gamma, scaled

        if i > 0 and gamma == self.Gamma:
            gamma = max(gamma - self.Gamma - 1, gamma - i - 1)
            i_back = max(0, i - self.Gamma)
            if i == len(req.vnfs):
                self.undo_chaining(req, i)
                i = i - 1
            for j in range(i_back, i + 1):
                self.undo(req, j)
            self.ilp_model.v_var[0][self.ilp_model.N_map[self.loc_of[i_back]], i_back].lb = 0.0
            self.ilp_model.v_var[0][self.ilp_model.N_map[self.loc_of[i_back]], i_back].ub = 0.0
            for j in range(i_back, i + 1):
                del self.loc_of[j]
            return True, i_back, first_bt, gamma, scaled

        self.cleanup(req)
        return False, i, first_bt, gamma, scaled

    def place(self, req, i):
        loc_pr = list()
        for n in self.ilp_model.Ec_id:
            loc_pr.append(self.ilp_model.v_var[0][n, i].x)
        best_loc = self.ilp_model.Ec_id[loc_pr.index(max(loc_pr))]
        self.ilp_model.v_var[0][best_loc, i].lb = 1.0
        self.loc_of[i] = self.ilp_model.N_map_inv[best_loc]
        self.my_net.g.nodes[self.loc_of[i]]["nd"].embed(req, i)
        req.used_servers.add(self.loc_of[i])

    def chain(self, req, i):
        pvn = req.entry_point if i == 0 else self.loc_of[i - 1]
        cdn = self.loc_of[i]
        if pvn != cdn:
            path_vals = list()
            for pth_id in range(len(self.my_net.paths_links[pvn][cdn])):
                a = self.ilp_model.q_var[0][self.ilp_model.N_map[pvn]][self.ilp_model.N_map[cdn]][pth_id, i].x
                path_vals.append(a)
            self.routing_paths[i] = path_vals.index(max(path_vals))
            for ll in self.my_net.paths_links[pvn][cdn][self.routing_paths[i]]:
                l_obj = self.my_net.g[ll[0]][ll[1]]["li"]
                l_obj.embed(req, i)
            self.ilp_model.q_var[0][self.ilp_model.N_map[pvn]][self.ilp_model.N_map[cdn]][self.routing_paths[i], i].lb = 1.0

    def round_dl(self, req, i):
        rounding_failed = False
        Rd_ei, _ = self.my_net.get_missing_layers(self.loc_of[i], req, i, req.tau1)
        for rr in Rd_ei:
            if i not in self.dl_paths:
                self.dl_paths[i] = dict()
            pth_pr = []
            pth_ids = range(len(self.my_net.paths_links[self.loc_of[i]][self.ilp_model.cloud_node]))
            for pth_id in pth_ids:
                a = self.ilp_model.w_var[0][self.ilp_model.N_map[self.loc_of[i]]][self.ilp_model.N_map[self.ilp_model.cloud_node]][pth_id, rr].x
                pth_pr.append(a)
            if sum(pth_pr) != 0.0:
                if sum(pth_pr) < 1.0:
                    pth_pr = [pr / sum(pth_pr) for pr in pth_pr]
                self.dl_paths[i][rr] = np.random.choice(a=pth_ids, p=pth_pr)
                self.ilp_model.w_var[0][self.ilp_model.N_map[self.loc_of[i]]][self.ilp_model.N_map[self.ilp_model.cloud_node]][self.dl_paths[i][self.ilp_model.R_id[rr]], rr].lb = 1.0
            else:
                print("one failed, no candidate path!")
                rounding_failed = True
                break
        return rounding_failed

    def do_download(self, req, i):
        self.my_net.g.nodes[self.loc_of[i]]["nd"].add_layer(req.vnfs[i].layers, req)
        self.downloads[i] = set()
        self.total_dl_vol[i] = 0
        if i in self.dl_paths:
            for rr in self.dl_paths[i]:
                self.total_dl_vol[i] = self.total_dl_vol[i] + self.Rvol[rr]
                layer_download = LayerDownload()
                self.downloads[i].add(layer_download)
                pp = self.dl_paths[i][rr]
                for tt in req.T1:
                    for ll in self.my_net.paths_links[self.loc_of[i]][self.ilp_model.cloud_node][pp]:
                        l_obj = self.my_net.g[ll[0]][ll[1]]["li"]
                        layer_download.add_data(tt, l_obj, self.Rvol[rr] / len(req.T1))

    def solve_single_relax(self, req):
        tr = TestResult()
        self.ilp_model = get_ilp([req], self.my_net, self.R, self.Rvol)

        self.ilp_model.m.update()
        for v in self.ilp_model.m.getVars():
            v.setAttr('vtype', 'C')
            v.lb = 0.0
            v.ub = 1.0

        self.ilp_model.m.setParam("LogToConsole", False)
        self.ilp_model.m.setParam("Threads" , 6)
        # m.setParam("TIME_LIMIT", 500)
        self.ilp_model.m.optimize()
        # m.write("out.lp")

        if self.ilp_model.m.status == GRB.INFEASIBLE:
            # m.computeIIS()
            # m.write("s_model.ilp")
            return tr.SF, 0, 0

        i = 0
        do_scale = True
        first_bt = self.Gamma
        gamma = self.Gamma
        self.loc_of[len(req.vnfs)] = req.entry_point
        while i <= len(req.vnfs):
            if i < len(req.vnfs):
                self.eliminate(req, i)
                link_time = dict()
                if self.bw_scaler < 1.0 and do_scale:
                    link_time = self.scale_links(req)
                self.ilp_model.m.optimize()
                if self.bw_scaler < 1.0 and do_scale:
                    self.rescale_links(req, link_time)
                if self.ilp_model.m.status == GRB.INFEASIBLE:
                    st, i, first_bt, gamma, do_scale = self.handle_backtrack(req, i, first_bt, gamma, do_scale)
                    if not st:
                        print("failed SF!")
                        return tr.SF, 0, 0
                    else:
                        continue
                self.place(req, i)
                self.chain(req, i)
                rounding_failed = self.round_dl(req, i)
                if not rounding_failed:
                    self.ilp_model.m.optimize()
                if rounding_failed or self.ilp_model.m.status == GRB.INFEASIBLE:
                    st, i, first_bt, gamma, do_scale = self.handle_backtrack(req, i, first_bt, gamma, do_scale)
                    if not st:
                        print("failed RF!")
                        return tr.RF, 0, 0
                    else:
                        continue
                self.do_download(req, i)
            else:
                self.chain(req, len(req.vnfs))
                self.ilp_model.m.optimize()
                if self.ilp_model.m.status == GRB.INFEASIBLE:
                    st, i, first_bt, gamma, do_scale = self.handle_backtrack(req, i, first_bt, gamma, do_scale)
                    if not st:
                        print("failed at last!")
                        return tr.SF, 0, 0
                    else:
                        continue
            do_scale = True
            i = i + 1
            gamma = min(self.Gamma, gamma+1)

        print("success!")
        for ii in range(len(req.vnfs)):
            if ii in self.loc_of:
                self.my_net.g.nodes[self.loc_of[ii]]["nd"].finalize_layer()
        return tr.SU, sum(self.total_dl_vol.values()), self.ilp_model.m.objVal
