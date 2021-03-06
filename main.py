# RCCO, (C) Mahdi Dolati. License: AGPLv3
import numpy as np

import heapq
import sys, getopt
from time import process_time, time

from sfc import SfcGenerator
from my_sys.net import NetGenerator
from solution import FfSolver
from solution import GreedySolver
from solution import GurobiSingleRelax
from solution import GurobiBatch
from constants import Const
from statistic_collector import StatCollector, Stat
from test import TestResult


def test(solver, reqs):
    solver.reset()
    tr = TestResult()
    accepted = 0.0
    layer_dl_vol = 0.0
    chain_bw_total = 0.0
    rev = 0.0
    sampling_rate = 1.0
    events = []
    counter = 1
    arrivals = 0
    vol_consumed = list()
    run_avg_admit = list()
    for s in reqs:
        heapq.heappush(events, (s.arrival_time, counter, "ARRIVAL", s))
        counter += 1
    while len(events) > 0:
        t, cnt, ev, s = heapq.heappop(events)
        if ev == "ARRIVAL":
            arrivals = arrivals + 1
            solver.pre_arrival_procedure(t)
            status, dl_vol, chain_bw = solver.solve(s, t, sampling_rate)
            tr.res_groups[status] = tr.res_groups[status] + 1
            if status == tr.SU:
                tr.accepted_reqs.append(s)
                solver.post_arrival_procedure(status, t, s)
                layer_dl_vol = layer_dl_vol + dl_vol
                chain_bw_total = chain_bw_total + chain_bw
                accepted = accepted + 1
                for ii in range(len(s.vnfs)+1):
                    rev = rev + s.vnf_in_rate(ii)
                heapq.heappush(events, (s.tau2+1, counter, "FINISH", s))
                counter += 1
            if arrivals % 40 == 0:
                vol_consumed.append(0.0 if accepted == 0 else layer_dl_vol / accepted)
                run_avg_admit.append(accepted / arrivals)
                print("{}, {}, {}".format(arrivals, run_avg_admit[-1], vol_consumed[-1]))
        elif ev == "FINISH":
            solver.handle_sfc_eviction(s, t)
    avg_rate = accepted / len(reqs)
    avg_dl = layer_dl_vol
    tr.revenue = rev
    tr.avg_admit = avg_rate
    tr.avg_dl = avg_dl
    tr.run_avg_dl = vol_consumed
    tr.run_avg_admit = run_avg_admit
    tr.chain_bw = chain_bw_total
    tr.avg_dl_per_acc = 0.0 if accepted == 0 else avg_dl / accepted
    print(tr.chain_bw)
    return tr


def batch_test(inter_arrival):
    np.random.seed(1)
    my_net = NetGenerator().get_g()
    req_nums = [6, 8, 10, 12, 14]
    sfc_gen = SfcGenerator(my_net, { 1: 1.0 }, 1.0)
    sfc_gen.print()
    # R_ids = [i for i in sfc_gen.layers]
    # R_vols = [sfc_gen.layers[i] for i in R_ids]
    # my_net.print()
    ACCEPT_RATIO = "Accept Ratio"
    DOWNLOAD_LAYER = "Download (MB)"
    RUNTIME = "Runtime (sec)"
    CHAIN_BW = "Chain (mbps)"
    REVENUE = "Revenue"
    solvers = [
        GurobiSingleRelax(2, 0.8, "popularity_learn"),
        GurobiBatch()
    ]
    stats = {ACCEPT_RATIO: Stat.MEAN_MODE,
             DOWNLOAD_LAYER: Stat.MEAN_MODE,
             CHAIN_BW: Stat.MEAN_MODE,
             RUNTIME: Stat.MEAN_MODE,
             REVENUE: Stat.MEAN_MODE}
    algs = [s.get_name() for s in solvers]
    stat_collector = StatCollector(algs, stats)
    #
    iterations = 3
    arrival_rate = 1.0 / inter_arrival
    for req_num in req_nums:
        run_name = "{:d}".format(req_num)
        print("run-name:", run_name)
        for itr in range(iterations):
            reqs = []
            t = 0
            np.random.seed(itr * 4321)
            for _ in range(req_num):
                reqs.append(sfc_gen.get_chain(t))
                t = t + int(np.ceil(np.random.exponential(1.0 / arrival_rate)))
                print(reqs[-1])
            #
            for solver in solvers:
                np.random.seed(itr * 1234)
                if solver.convert_layer:
                    R_ids, R_vols = solver.do_convert_no_share(reqs)
                else:
                    R_ids = [i for i in sfc_gen.layers]
                    R_vols = sfc_gen.layers
                solver.set_env(my_net, R_ids, R_vols)
                t1 = time()
                if solver.batch:
                    solver.reset()
                    tr = solver.solve_batch2(my_net, sfc_gen.vnfs_list, reqs)
                else:
                    tr = test(solver, reqs)
                    reqs = tr.accepted_reqs
                    print("Solver: {} got {} out of {}".format(solver.get_name(), tr.avg_admit, req_num))
                t2 = time()
                stat_collector.add_stat(solver.get_name(), ACCEPT_RATIO, run_name, tr.avg_admit)
                stat_collector.add_stat(solver.get_name(), DOWNLOAD_LAYER, run_name, tr.avg_dl)
                stat_collector.add_stat(solver.get_name(), RUNTIME, run_name, t2-t1)
                stat_collector.add_stat(solver.get_name(), CHAIN_BW, run_name, tr.chain_bw)
                stat_collector.add_stat(solver.get_name(), REVENUE, run_name, tr.revenue)

    machine_id = "ut"
    fig_test_id = "{}_batch".format(machine_id)
    inter_arrival = str(inter_arrival).replace(".", "_")
    fig_2 = './result/{}_accept_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', req_nums, 0, ACCEPT_RATIO, algs, 'Share Percentage', ACCEPT_RATIO)

    fig_2 = './result/{}_dl_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', req_nums, 0, DOWNLOAD_LAYER, algs, 'Share Percentage', DOWNLOAD_LAYER)

    fig_3 = './result/{}_time_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_3 + '.txt', req_nums, 0, RUNTIME, algs, 'Share Percentage', RUNTIME)

    fig_4 = './result/{}_chain_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_4 + '.txt', req_nums, 0, CHAIN_BW, algs, 'Chaining BW', CHAIN_BW)

    fig_5 = './result/{}_rev_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_5 + '.txt', req_nums, 0, REVENUE, algs, 'Revenue', REVENUE)


def optimal_test(inter_arrival, scale_bw=False):
    np.random.seed(1)
    if scale_bw:
        print("Using scaled bandwidth!")
        Const.LINK_BW = [10000, 20000]
    my_net = NetGenerator().get_g()
    req_nums = [6, 8, 10, 12, 14]
    sfc_gen = SfcGenerator(my_net, { 1: 1.0 }, 1.0)
    sfc_gen.print()
    # R_ids = [i for i in sfc_gen.layers]
    # R_vols = [sfc_gen.layers[i] for i in R_ids]
    # my_net.print()
    ACCEPT_RATIO = "Accept Ratio"
    DOWNLOAD_LAYER = "Download (MB)"
    RUNTIME = "Runtime (sec)"
    CHAIN_BW = "Chain (mbps)"
    REVENUE = "Revenue"
    solvers = [
        GurobiSingleRelax(2, 0.8, "popularity_learn"),
        GurobiBatch()
    ]
    stats = {ACCEPT_RATIO: Stat.MEAN_MODE,
             DOWNLOAD_LAYER: Stat.MEAN_MODE,
             CHAIN_BW: Stat.MEAN_MODE,
             RUNTIME: Stat.MEAN_MODE,
             REVENUE: Stat.MEAN_MODE}
    algs = [s.get_name() for s in solvers]
    stat_collector = StatCollector(algs, stats)
    #
    iterations = 3
    arrival_rate = 1.0 / inter_arrival
    for req_num in req_nums:
        run_name = "{:d}".format(req_num)
        print("run-name:", run_name)
        for itr in range(iterations):
            reqs = []
            t = 0
            np.random.seed(itr * 4321)
            for _ in range(req_num):
                reqs.append(sfc_gen.get_chain(t))
                t = t + int(np.ceil(np.random.exponential(1.0 / arrival_rate)))
                print(reqs[-1])
            #
            for solver in solvers:
                np.random.seed(itr * 1234)
                if solver.convert_layer:
                    R_ids, R_vols = solver.do_convert_no_share(reqs)
                else:
                    R_ids = [i for i in sfc_gen.layers]
                    R_vols = sfc_gen.layers
                solver.set_env(my_net, R_ids, R_vols)
                t1 = process_time()
                if solver.batch:
                    solver.reset()
                    tr = solver.solve_batch(my_net, sfc_gen.vnfs_list, reqs)
                else:
                    tr = test(solver, reqs)
                    print("Solver: {} got {} out of {}".format(solver.get_name(), tr.avg_admit, req_num))
                t2 = process_time()
                stat_collector.add_stat(solver.get_name(), ACCEPT_RATIO, run_name, tr.avg_admit)
                stat_collector.add_stat(solver.get_name(), DOWNLOAD_LAYER, run_name, tr.avg_dl)
                stat_collector.add_stat(solver.get_name(), RUNTIME, run_name, t2-t1)
                stat_collector.add_stat(solver.get_name(), CHAIN_BW, run_name, tr.chain_bw)
                stat_collector.add_stat(solver.get_name(), REVENUE, run_name, tr.revenue)

    machine_id = "ut"
    fig_test_id = "{}_optimal_bw{}".format(machine_id, scale_bw)
    inter_arrival = str(inter_arrival).replace(".", "_")
    fig_2 = './result/{}_accept_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', req_nums, 0, ACCEPT_RATIO, algs, 'Share Percentage', ACCEPT_RATIO)

    fig_2 = './result/{}_dl_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', req_nums, 0, DOWNLOAD_LAYER, algs, 'Share Percentage', DOWNLOAD_LAYER)

    fig_3 = './result/{}_time_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_3 + '.txt', req_nums, 0, RUNTIME, algs, 'Share Percentage', RUNTIME)

    fig_4 = './result/{}_chain_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_4 + '.txt', req_nums, 0, CHAIN_BW, algs, 'Chaining BW', CHAIN_BW)

    fig_5 = './result/{}_rev_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_5 + '.txt', req_nums, 0, REVENUE, algs, 'Revenue', REVENUE)


def scaling_test(inter_arrival):
    np.random.seed(1)
    my_net = NetGenerator().get_g()
    req_nums = [50]
    sfc_gen = SfcGenerator(my_net, {1: 1.0}, 1.0)
    sfc_gen.print()
    # R_ids = [i for i in sfc_gen.layers]
    # R_vols = [sfc_gen.layers[i] for i in R_ids]
    # my_net.print()
    ACCEPT_RATIO = "Accept Ratio"
    DOWNLOAD_LAYER = "Download (MB)"
    RUNTIME = "Runtime (sec)"
    CHAIN_BW = "Chain (mbps)"
    REVENUE = "Revenue"
    scaling_factors = [1.0, 0.9, 0.8, 0.7]
    stats = {ACCEPT_RATIO: Stat.MEAN_MODE,
             DOWNLOAD_LAYER: Stat.MEAN_MODE,
             CHAIN_BW: Stat.MEAN_MODE,
             RUNTIME: Stat.MEAN_MODE,
             REVENUE: Stat.MEAN_MODE}
    algs = list()
    solvers = []
    for sf in scaling_factors:
        solver = GurobiSingleRelax(0, sf, "popularity_learn")
        rgx = "{:.1f}".format(solver.bw_scaler)
        algs.append(solver.get_name() + rgx)
        solvers.append(solver)
    stat_collector = StatCollector(algs, stats)
    GROUP = "GROUP"
    stats2 = {GROUP: Stat.SUM_MODE}
    tr2 = TestResult()
    algs2 = [tr2.SU, tr2.SF, tr2.RF]
    stat_collector2 = StatCollector(algs2, stats2)
    #
    iterations = 3
    arrival_rate = 1.0 / inter_arrival
    for req_num in req_nums:
        run_name = "{:d}".format(req_num)
        print("run-name:", run_name)
        for itr in range(iterations):
            np.random.seed(itr * 1234)
            reqs = []
            t = 0
            for _ in range(req_num):
                reqs.append(sfc_gen.get_chain(t))
                t = t + int(np.ceil(np.random.exponential(1.0 / arrival_rate)))
                print(reqs[-1])
            for solver in solvers:
                np.random.seed(itr * 4321)
                if solver.convert_layer:
                    R_ids, R_vols = solver.do_convert_no_share(reqs)
                else:
                    R_ids = [i for i in sfc_gen.layers]
                    R_vols = sfc_gen.layers
                solver.set_env(my_net, R_ids, R_vols)
                t1 = process_time()
                if solver.batch:
                    tr = solver.solve_batch(my_net, sfc_gen.vnfs_list, reqs)
                else:
                    tr = test(solver, reqs)
                    print("Solver: {} got {}".format(solver.get_name(), tr))
                t2 = process_time()
                rgx = "{:.1f}".format(solver.bw_scaler)
                stat_collector.add_stat(solver.get_name()+rgx, ACCEPT_RATIO, run_name, tr.avg_admit)
                stat_collector.add_stat(solver.get_name()+rgx, DOWNLOAD_LAYER, run_name, tr.avg_dl)
                stat_collector.add_stat(solver.get_name()+rgx, RUNTIME, run_name, t2 - t1)
                stat_collector.add_stat(solver.get_name()+rgx, CHAIN_BW, run_name, tr.chain_bw)
                stat_collector.add_stat(solver.get_name()+rgx, REVENUE, run_name, tr.revenue)
                for rg in tr.res_groups:
                    rgx = "{:.1f}".format(solver.bw_scaler)
                    stat_collector2.add_stat(rg, GROUP, rgx, tr.res_groups[rg])

    machine_id = "ut"
    fig_test_id = "{}_scaling".format(machine_id)
    inter_arrival = str(inter_arrival).replace(".", "_")
    fig_2 = './result/{}_accept_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', req_nums, 0, ACCEPT_RATIO, algs, 'Share Percentage', ACCEPT_RATIO)

    fig_2 = './result/{}_dl_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', req_nums, 0, DOWNLOAD_LAYER, algs, 'Share Percentage', DOWNLOAD_LAYER)

    fig_3 = './result/{}_time_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_3 + '.txt', req_nums, 0, RUNTIME, algs, 'Share Percentage', RUNTIME)

    fig_4 = './result/{}_chain_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_4 + '.txt', req_nums, 0, CHAIN_BW, algs, 'Chaining BW', CHAIN_BW)

    fig_5 = './result/{}_rev_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_5 + '.txt', req_nums, 0, REVENUE, algs, 'Revenue', REVENUE)

    fig_6 = './result/{}_ss_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector2.write_to_file(fig_6 + '.txt', scaling_factors, 0, GROUP, algs2, 'GROUP', GROUP)


def backtrack_test(inter_arrival):
    np.random.seed(1)
    my_net = NetGenerator().get_g()
    req_nums = [50]
    sfc_gen = SfcGenerator(my_net, {1: 1.0}, 1.0)
    sfc_gen.print()
    # R_ids = [i for i in sfc_gen.layers]
    # R_vols = [sfc_gen.layers[i] for i in R_ids]
    # my_net.print()
    ACCEPT_RATIO = "Accept Ratio"
    DOWNLOAD_LAYER = "Download (MB)"
    RUNTIME = "Runtime (sec)"
    CHAIN_BW = "Chain (mbps)"
    REVENUE = "Revenue"
    backtrack_vals = [0, 1, 2, 3]
    solvers = []
    algs = []
    for bv in backtrack_vals:
        solvers.append(GurobiSingleRelax(bv, 1.0, "popularity_learn"))
        rgx = "{:d}".format(solvers[-1].Gamma)
        algs.append(solvers[-1].get_name()+rgx)
    stats = {ACCEPT_RATIO: Stat.MEAN_MODE,
             DOWNLOAD_LAYER: Stat.MEAN_MODE,
             CHAIN_BW: Stat.MEAN_MODE,
             RUNTIME: Stat.MEAN_MODE,
             REVENUE: Stat.MEAN_MODE}
    stat_collector = StatCollector(algs, stats)
    GROUP = "GROUP"
    stats2 = {GROUP: Stat.SUM_MODE}
    tr2 = TestResult()
    algs2 = [tr2.SU, tr2.SF, tr2.RF]
    stat_collector2 = StatCollector(algs2, stats2)
    #
    iterations = 3
    arrival_rate = 1.0 / inter_arrival
    for req_num in req_nums:
        run_name = "{:d}".format(req_num)
        print("run-name:", run_name)
        for itr in range(iterations):
            reqs = []
            t = 0
            np.random.seed(itr * 4321)
            for _ in range(req_num):
                reqs.append(sfc_gen.get_chain(t))
                t = t + int(np.ceil(np.random.exponential(1.0 / arrival_rate)))
                print(reqs[-1])
            #
            for solver in solvers:
                np.random.seed(itr * 1234)
                if solver.convert_layer:
                    R_ids, R_vols = solver.do_convert_no_share(reqs)
                else:
                    R_ids = [i for i in sfc_gen.layers]
                    R_vols = sfc_gen.layers
                solver.set_env(my_net, R_ids, R_vols)
                t1 = process_time()
                if solver.batch:
                    tr = solver.solve_batch(my_net, sfc_gen.vnfs_list, reqs)
                else:
                    tr = test(solver, reqs)
                    print("Solver: {} got {}".format(solver.get_name(), tr))
                t2 = process_time()
                rgx = "{:d}".format(solver.Gamma)
                stat_collector.add_stat(solver.get_name()+rgx, ACCEPT_RATIO, run_name, tr.avg_admit)
                stat_collector.add_stat(solver.get_name()+rgx, DOWNLOAD_LAYER, run_name, tr.avg_dl)
                stat_collector.add_stat(solver.get_name()+rgx, RUNTIME, run_name, t2 - t1)
                stat_collector.add_stat(solver.get_name()+rgx, CHAIN_BW, run_name, tr.chain_bw)
                stat_collector.add_stat(solver.get_name()+rgx, REVENUE, run_name, tr.revenue)
                #if solver.get_name()[0:2] == "Gr":
                for rg in tr.res_groups:
                    rgx = "{:d}".format(solver.Gamma)
                    stat_collector2.add_stat(rg, GROUP, rgx, tr.res_groups[rg])

    machine_id = "ut"
    fig_test_id = "{}_backtrack".format(machine_id)
    inter_arrival = str(inter_arrival).replace(".", "_")
    fig_2 = './result/{}_accept_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', req_nums, 0, ACCEPT_RATIO, algs, 'Share Percentage', ACCEPT_RATIO)

    fig_2 = './result/{}_dl_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', req_nums, 0, DOWNLOAD_LAYER, algs, 'Share Percentage', DOWNLOAD_LAYER)

    fig_3 = './result/{}_time_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_3 + '.txt', req_nums, 0, RUNTIME, algs, 'Share Percentage', RUNTIME)

    fig_4 = './result/{}_chain_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_4 + '.txt', req_nums, 0, CHAIN_BW, algs, 'Chaining BW', CHAIN_BW)

    fig_5 = './result/{}_rev_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_5 + '.txt', req_nums, 0, REVENUE, algs, 'Revenue', REVENUE)

    fig_6 = './result/{}_ss_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector2.write_to_file(fig_6 + '.txt', backtrack_vals, 0, GROUP, algs2, 'GROUP', GROUP)


def share_percentage_test(inter_arrival):
    np.random.seed(1)
    my_net = NetGenerator().get_g()
    # my_net.print()
    ACCEPT_RATIO = "Accept Ratio"
    DOWNLOAD_LAYER = "Download (MB)"
    RUNTIME = "Runtime (sec)"
    CHAIN_BW = "Chain (mbps)"
    REVENUE = "Revenue"
    DL_ACC = "DL_ACC"
    solvers = [
        FfSolver(),
        GreedySolver(),
        GurobiSingleRelax(2, 0.8, "popularity_learn", False),
        GurobiSingleRelax(2, 0.8, "popularity_learn", True),
    ]
    stats = {ACCEPT_RATIO: Stat.MEAN_MODE,
             DOWNLOAD_LAYER: Stat.MEAN_MODE,
             CHAIN_BW: Stat.MEAN_MODE,
             RUNTIME: Stat.MEAN_MODE,
             REVENUE: Stat.MEAN_MODE,
             DL_ACC: Stat.MEAN_MODE}
    algs = [s.get_name() for s in solvers]
    stat_collector = StatCollector(algs, stats)
    #
    iterations = 3
    arrival_rate = 1.0 / inter_arrival
    layer_magnitude = [5, 10, 15, 20]
    # layer_magnitude = [5]
    vnf_num = 10
    for i in range(len(layer_magnitude)):
        np.random.seed(i * 100)

        Const.VNF_NUM = vnf_num
        Const.LAYER_NUM = vnf_num * layer_magnitude[i]

        sfc_gen = SfcGenerator(my_net, {1: 1.0}, 1.0)

        run_name = "{}".format(layer_magnitude[i])
        print("run-name:", run_name)
        for itr in range(iterations):
            reqs = []
            req_num = 50
            # req_num = 15
            t = 0
            np.random.seed(itr * 4321)
            for _ in range(req_num):
                reqs.append(sfc_gen.get_chain(t))
                t = t + int(np.ceil(np.random.exponential(1.0 / arrival_rate)))
            for solver in solvers:
                np.random.seed(itr * 1234)
                if solver.convert_layer:
                    R_ids, R_vols = solver.do_convert_no_share(reqs)
                else:
                    R_ids = [i for i in sfc_gen.layers]
                    R_vols = sfc_gen.layers
                    # R_ids, R_vols = solver.get_Rid_vol(reqs)
                solver.set_env(my_net, R_ids, R_vols)
                t1 = process_time()
                if solver.batch:
                    tr = solver.solve_batch(my_net, sfc_gen.vnfs_list, reqs)
                else:
                    tr = test(solver, reqs)
                    print("Solver: {} got {}".format(solver.get_name(), tr))
                t2 = process_time()
                stat_collector.add_stat(solver.get_name(), ACCEPT_RATIO, run_name, tr.avg_admit)
                stat_collector.add_stat(solver.get_name(), DOWNLOAD_LAYER, run_name, tr.avg_dl)
                stat_collector.add_stat(solver.get_name(), RUNTIME, run_name, t2 - t1)
                stat_collector.add_stat(solver.get_name(), CHAIN_BW, run_name, tr.chain_bw)
                stat_collector.add_stat(solver.get_name(), REVENUE, run_name, tr.revenue)
                stat_collector.add_stat(solver.get_name(), DL_ACC, run_name, tr.avg_dl_per_acc)

    machine_id = "ut"
    fig_test_id = "{}_share".format(machine_id)
    inter_arrival = str(inter_arrival).replace(".", "_")
    fig_2 = './result/{}_accept_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', layer_magnitude, 0, ACCEPT_RATIO, algs, 'Share Percentage', ACCEPT_RATIO)

    fig_2 = './result/{}_dl_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', layer_magnitude, 0, DOWNLOAD_LAYER, algs, 'Share Percentage', DOWNLOAD_LAYER)

    fig_3 = './result/{}_time_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_3 + '.txt', layer_magnitude, 0, RUNTIME, algs, 'Share Percentage', RUNTIME)

    fig_4 = './result/{}_chain_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_4 + '.txt', layer_magnitude, 0, CHAIN_BW, algs, 'Chaining BW', CHAIN_BW)

    fig_5 = './result/{}_rev_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_5 + '.txt', layer_magnitude, 0, REVENUE, algs, 'Revenue', REVENUE)

    fig_6 = './result/{}_dla_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_6 + '.txt', layer_magnitude, 0, DL_ACC, algs, 'Revenue', DL_ACC)


def layer_num_test(inter_arrival):
    np.random.seed(1)
    my_net = NetGenerator().get_g()
    # my_net.print()
    ACCEPT_RATIO = "Accept Ratio"
    DOWNLOAD_LAYER = "Download (MB)"
    RUNTIME = "Runtime (sec)"
    CHAIN_BW = "Chain (mbps)"
    REVENUE = "Revenue"
    DL_ACC = "DL_ACC"
    solvers = [
        FfSolver(),
        GreedySolver(),
        GurobiSingleRelax(2, 0.8, "popularity_learn", False),
        GurobiSingleRelax(2, 0.8, "popularity_learn", True),
    ]
    stats = {ACCEPT_RATIO: Stat.MEAN_MODE,
             DOWNLOAD_LAYER: Stat.MEAN_MODE,
             CHAIN_BW: Stat.MEAN_MODE,
             RUNTIME: Stat.MEAN_MODE,
             REVENUE: Stat.MEAN_MODE,
             DL_ACC: Stat.MEAN_MODE}
    algs = [s.get_name() for s in solvers]
    stat_collector = StatCollector(algs, stats)
    #
    iterations = 3
    arrival_rate = 1.0 / inter_arrival
    layer_num = [2, 6, 10, 14]
    vnf_size = 420
    for i in range(len(layer_num)):
        np.random.seed(i * 100)

        Const.VNF_LAYER = [layer_num[i], layer_num[i]+1]
        Const.LAYER_SIZE = [vnf_size/layer_num[i], (vnf_size/layer_num[i])+1]
        sfc_gen = SfcGenerator(my_net, {1: 1.0}, 1.0)

        run_name = "{}".format(layer_num[i])
        print("run-name:", run_name)
        for itr in range(iterations):
            reqs = []
            req_num = 50
            t = 0
            np.random.seed(itr * 4321)
            for _ in range(req_num):
                reqs.append(sfc_gen.get_chain(t))
                t = t + int(np.ceil(np.random.exponential(1.0 / arrival_rate)))
            for solver in solvers:
                np.random.seed(itr * 1234)
                if solver.convert_layer:
                    R_ids, R_vols = solver.do_convert_no_share(reqs)
                else:
                    R_ids = [i for i in sfc_gen.layers]
                    R_vols = sfc_gen.layers
                    # R_ids, R_vols = solver.get_Rid_vol(reqs)
                solver.set_env(my_net, R_ids, R_vols)
                t1 = process_time()
                if solver.batch:
                    tr = solver.solve_batch(my_net, sfc_gen.vnfs_list, reqs)
                else:
                    tr = test(solver, reqs)
                    print("Solver: {} got {}".format(solver.get_name(), tr))
                t2 = process_time()
                stat_collector.add_stat(solver.get_name(), ACCEPT_RATIO, run_name, tr.avg_admit)
                stat_collector.add_stat(solver.get_name(), DOWNLOAD_LAYER, run_name, tr.avg_dl)
                stat_collector.add_stat(solver.get_name(), RUNTIME, run_name, t2 - t1)
                stat_collector.add_stat(solver.get_name(), CHAIN_BW, run_name, tr.chain_bw)
                stat_collector.add_stat(solver.get_name(), REVENUE, run_name, tr.revenue)
                stat_collector.add_stat(solver.get_name(), DL_ACC, run_name, tr.avg_dl_per_acc)

    machine_id = "ut"
    fig_test_id = "{}_layer_num".format(machine_id)
    inter_arrival = str(inter_arrival).replace(".", "_")
    fig_2 = './result/{}_accept_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', layer_num, 0, ACCEPT_RATIO, algs, 'Share Percentage', ACCEPT_RATIO)

    fig_2 = './result/{}_dl_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', layer_num, 0, DOWNLOAD_LAYER, algs, 'Share Percentage', DOWNLOAD_LAYER)

    fig_3 = './result/{}_time_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_3 + '.txt', layer_num, 0, RUNTIME, algs, 'Share Percentage', RUNTIME)

    fig_4 = './result/{}_chain_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_4 + '.txt', layer_num, 0, CHAIN_BW, algs, 'Chaining BW', CHAIN_BW)

    fig_5 = './result/{}_rev_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_5 + '.txt', layer_num, 0, REVENUE, algs, 'Revenue', REVENUE)

    fig_6 = './result/{}_dla_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_6 + '.txt', layer_num, 0, DL_ACC, algs, 'Revenue', DL_ACC)


def no_share_test(inter_arrival):
    np.random.seed(1)
    my_net = NetGenerator().get_g()
    # my_net.print()
    ACCEPT_RATIO = "Accept Rate"
    DOWNLOAD_LAYER = "Download (Mb)"
    RUNTIME = "Runtime (sec)"
    CHAIN_BW = "Chain (mbps)"
    REVENUE = "Revenue (mbps)"
    DL_ACC = "Download (Mb)"
    solvers = [
        FfSolver(),
        GreedySolver(),
        GurobiSingleRelax(2, 0.8, "popularity_learn", False),
        GurobiSingleRelax(2, 0.8, "popularity_learn", True),
    ]
    stats = {ACCEPT_RATIO: Stat.MEAN_MODE,
             DOWNLOAD_LAYER: Stat.MEAN_MODE,
             CHAIN_BW: Stat.MEAN_MODE,
             RUNTIME: Stat.MEAN_MODE,
             REVENUE: Stat.MEAN_MODE,
             DL_ACC: Stat.MEAN_MODE}
    algs = [s.get_name() for s in solvers]
    stat_collector = StatCollector(algs, stats)
    #
    iterations = 3
    arrival_rate = 1.0 / inter_arrival
    layer_magnitude = [20, 15, 10, 5]
    # layer_magnitude = [5]
    vnf_num = 10
    for i in range(len(layer_magnitude)):
        np.random.seed(i * 100)

        Const.VNF_NUM = vnf_num
        Const.LAYER_NUM = vnf_num * layer_magnitude[i]

        sfc_gen = SfcGenerator(my_net, {1: 1.0}, 1.0)

        run_name = "{}".format(layer_magnitude[i])
        print("run-name:", run_name)
        for itr in range(iterations):
            reqs = []
            req_num = 50
            # req_num = 15
            t = 0
            np.random.seed(itr * 4321)
            for _ in range(req_num):
                reqs.append(sfc_gen.get_chain(t))
                t = t + int(np.ceil(np.random.exponential(1.0 / arrival_rate)))
            for solver in solvers:
                np.random.seed(itr * 1234)
                if solver.convert_layer:
                    R_ids, R_vols = solver.do_convert_no_share(reqs)
                else:
                    R_ids = [i for i in sfc_gen.layers]
                    R_vols = sfc_gen.layers
                    # R_ids, R_vols = solver.get_Rid_vol(reqs)
                solver.set_env(my_net, R_ids, R_vols)
                t1 = process_time()
                if solver.batch:
                    tr = solver.solve_batch(my_net, sfc_gen.vnfs_list, reqs)
                else:
                    tr = test(solver, reqs)
                    print("Solver: {} got {}".format(solver.get_name(), tr))
                t2 = process_time()
                stat_collector.add_stat(solver.get_name(), ACCEPT_RATIO, run_name, tr.avg_admit)
                stat_collector.add_stat(solver.get_name(), DOWNLOAD_LAYER, run_name, tr.avg_dl)
                stat_collector.add_stat(solver.get_name(), RUNTIME, run_name, t2 - t1)
                stat_collector.add_stat(solver.get_name(), CHAIN_BW, run_name, tr.chain_bw)
                stat_collector.add_stat(solver.get_name(), REVENUE, run_name, tr.revenue)
                stat_collector.add_stat(solver.get_name(), DL_ACC, run_name, tr.avg_dl_per_acc)

    machine_id = "ut"
    fig_test_id = "{}_no_share".format(machine_id)
    inter_arrival = str(inter_arrival).replace(".", "_")
    fig_2 = './result/{}_accept_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', layer_magnitude, 0, ACCEPT_RATIO, algs, 'Share Percentage',
                                 ACCEPT_RATIO)

    fig_2 = './result/{}_dl_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', layer_magnitude, 0, DOWNLOAD_LAYER, algs, 'Share Percentage',
                                 DOWNLOAD_LAYER)

    fig_3 = './result/{}_time_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_3 + '.txt', layer_magnitude, 0, RUNTIME, algs, 'Share Percentage', RUNTIME)

    fig_4 = './result/{}_chain_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_4 + '.txt', layer_magnitude, 0, CHAIN_BW, algs, 'Chaining BW', CHAIN_BW)

    fig_5 = './result/{}_rev_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_5 + '.txt', layer_magnitude, 0, REVENUE, algs, 'Revenue', REVENUE)

    fig_6 = './result/{}_dla_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_6 + '.txt', layer_magnitude, 0, DL_ACC, algs, 'Revenue', DL_ACC)


def test_qlearning(inter_arrival):
    np.random.seed(1)
    Const.LINK_BW = [100, 1000]
    ACCEPT_RATIO = "Accept Ratio"
    DOWNLOAD_LAYER = "Download (MB)"
    STEP_DL_LAYER = "Rung Download (MB)"
    RUN_AVG_ADMIT = "Run Admit"
    CHAIN_BW = "Chain (mbps)"
    my_net = NetGenerator().get_g()
    sfc_gen = SfcGenerator(my_net, { 1: 1.0 }, 1.0)
    # R_ids = [i for i in sfc_gen.layers]
    # R_vols = [sfc_gen.layers[i] for i in R_ids]
    solvers = [
        GurobiSingleRelax(2, 0.9, "popularity_learn"),
        GurobiSingleRelax(2, 0.9, "default"),
    ]
    stats = {
        ACCEPT_RATIO: Stat.MEAN_MODE,
        DOWNLOAD_LAYER: Stat.MEAN_MODE,
        CHAIN_BW: Stat.MEAN_MODE
    }
    stats2 = {
        STEP_DL_LAYER: Stat.MEAN_MODE,
        RUN_AVG_ADMIT: Stat.MEAN_MODE
    }
    algs = [s.get_name()+s.eviction_strategy for s in solvers]
    stat_collector = StatCollector(algs, stats)
    stat_collector2 = StatCollector(algs, stats2)
    arrival_rate = 1.0 / inter_arrival
    run_name = "1"
    iterations = 1
    x_axis = [1]
    x_axis2 = []
    for itr in range(iterations):
        req_num = 1000
        t = 0
        reqs = []
        np.random.seed(itr * 4321)
        for _ in range(req_num):
            reqs.append(sfc_gen.get_chain(t))
            t = t + int(np.ceil(np.random.exponential(1.0 / arrival_rate)))
        for solver in solvers:
            np.random.seed(itr * 1234)
            if solver.convert_layer:
                R_ids, R_vols = solver.do_convert_no_share(reqs)
            else:
                R_ids = [i for i in sfc_gen.layers]
                R_vols = sfc_gen.layers
            solver.set_env(my_net, R_ids, R_vols)
            tr = test(solver, reqs)
            rgx = solver.eviction_strategy
            print("{}-Solver: {} got {} out of {}, dl_vol {}".format(itr, solver.get_name()+rgx, tr.avg_admit, req_num, tr.avg_dl))
            x_axis2 = list(range(len(tr.run_avg_dl)))
            for i in range(len(tr.run_avg_dl)):
                stat_collector2.add_stat(solver.get_name()+rgx, STEP_DL_LAYER, str(i), tr.run_avg_dl[i])
                stat_collector2.add_stat(solver.get_name()+rgx, RUN_AVG_ADMIT, str(i), tr.run_avg_admit[i])
            stat_collector.add_stat(solver.get_name()+rgx, ACCEPT_RATIO, run_name, tr.avg_admit)
            stat_collector.add_stat(solver.get_name()+rgx, DOWNLOAD_LAYER, run_name, tr.avg_dl)
            stat_collector.add_stat(solver.get_name()+rgx, CHAIN_BW, run_name, tr.chain_bw)

    machine_id = "ut"
    fig_test_id = "{}_eviction".format(machine_id)
    inter_arrival = str(inter_arrival).replace(".", "_")
    fig_2 = './result/{}_accept_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', x_axis, 0, ACCEPT_RATIO, algs, 'No. of Layers', ACCEPT_RATIO)

    fig_2 = './result/{}_dl_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', x_axis, 0, DOWNLOAD_LAYER, algs, 'No. of Layers', DOWNLOAD_LAYER)

    fig_2 = './result/{}_cg_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector2.write_to_file(fig_2 + '.txt', x_axis2, 0, STEP_DL_LAYER, algs, 'Steps', STEP_DL_LAYER)

    fig_2 = './result/{}_ra_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector2.write_to_file(fig_2 + '.txt', x_axis2, 0, RUN_AVG_ADMIT, algs, 'Steps', RUN_AVG_ADMIT)

    fig_2 = './result/{}_chain_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', x_axis, 0, CHAIN_BW, algs, 'Chaining BW', CHAIN_BW)


if __name__ == "__main__":
    my_argv = sys.argv[1:]
    opts, args = getopt.getopt(my_argv, "", ["inter-arrival=", "test-type=", "scale-bw="])
    test_type = "share"
    ia = 1.0
    scale_bw = False
    for opt, arg in opts:
        if opt in ("--inter-arrival",):
            ia = float(arg)
        elif opt in ("--test-type",):
            test_type = arg
        elif opt in ("--scale-bw",):
            scale_bw = arg == "True"
    if test_type == "scaling" or test_type == "all":
        print("running scaling because of {}".format(test_type))
        scaling_test(ia)
    if test_type == "layer" or test_type == "all":
        print("running layer because of {}".format(test_type))
        layer_num_test(ia)
    if test_type == "backtrack" or test_type == "all":
        print("running backtrack because of {}".format(test_type))
        backtrack_test(ia)
    if test_type == "share" or test_type == "all":
        print("running share because of {}".format(test_type))
        share_percentage_test(ia)
    if test_type == "optimal" or test_type == "all":
        print("running optimal because of {}".format(test_type))
        optimal_test(ia, scale_bw)
    if test_type == "learning" or test_type == "all":
        print("running qlearn because of {}".format(test_type))
        test_qlearning(ia)
    if test_type == "batch" or test_type == "all":
        print("running batch because of {}".format(test_type))
        batch_test(ia)
    if test_type == "noshare" or test_type == "all":
        print("running no share because of {}".format(test_type))
        no_share_test(ia)
