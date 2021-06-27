from sfc import SfcGenerator
from net import NetGenerator
from solution import NoShareSolver, ShareSolver, PopularitySolver
from constants import Const
from statistic_collector import StatCollector, Stat
import heapq
import numpy as np
import sys, getopt


def test(solver, reqs):
    solver.reset()
    rate = 0.0
    layer_dl_vol = 0.0
    sampling_rate = 1.0
    events = []
    counter = 1
    for s in reqs:
        heapq.heappush(events, (s.arrival_time, counter, "ARRIVAL", s))
        counter += 1
    while len(events) > 0:
        t, cnt, ev, s = heapq.heappop(events)
        if ev == "ARRIVAL":
            solver.pre_arrival_procedure(t)
            status, dl_vol = solver.solve(s, t, sampling_rate)
            if status:
                layer_dl_vol = layer_dl_vol + dl_vol
                rate = rate + 1
                heapq.heappush(events, (s.tau2+1, counter, "FINISH", s))
                counter += 1
        elif ev == "FINISH":
            solver.handle_sfc_eviction(s)
    avg_rate = rate / len(reqs)
    avg_dl = layer_dl_vol / rate if rate > 0 else 0
    return avg_rate, avg_dl


def share_percentage_test(inter_arrival):
    np.random.seed(1)
    my_net = NetGenerator().get_g()
    ACCEPT_RATIO = "Accept Ratio"
    DOWNLOAD_LAYER = "Download (MB)"
    solvers = [
        NoShareSolver(my_net, 0),
        ShareSolver(my_net, 0),
        ShareSolver(my_net, 1),
        ShareSolver(my_net, 2),
        PopularitySolver(my_net, 1),
        PopularitySolver(my_net, 2)
    ]
    stats = {ACCEPT_RATIO: Stat.MEAN_MODE, DOWNLOAD_LAYER: Stat.MEAN_MODE}
    algs = [s.get_name() for s in solvers]
    stat_collector = StatCollector(algs, stats)
    #
    iterations = 5
    arrival_rate = 1.0 / inter_arrival
    n_share_ps = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    share_percentages = []
    Const.VNF_LAYER = [3, 6]
    Const.LAYER_SIZE = [60, 101]
    for i in range(len(n_share_ps)):
        np.random.seed(i * 100)
        n_share_p = n_share_ps[i]
        x = n_share_p
        share_percentages.append(x)
        sfc_gen = SfcGenerator(my_net, n_share_p)
        run_name = "{:.2f}".format(x)
        print("run-name:", run_name)
        for itr in range(iterations):
            reqs = []
            req_num = 150
            t = 0
            for _ in range(req_num):
                reqs.append(sfc_gen.get_chain(t))
                t = t + int(np.ceil(np.random.exponential(1.0 / arrival_rate)))
            for solver in solvers:
                np.random.seed(itr * 1234)
                res, dl_vol = test(solver, reqs)
                stat_collector.add_stat(solver.get_name(), ACCEPT_RATIO, run_name, res)
                stat_collector.add_stat(solver.get_name(), DOWNLOAD_LAYER, run_name, dl_vol)

    machine_id = "ut"
    fig_test_id = "{}_share".format(machine_id)
    fig_2 = './result/{}_accept_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', share_percentages, 0, ACCEPT_RATIO, algs, 'Share Percentage', ACCEPT_RATIO)

    fig_2 = './result/{}_dl_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', share_percentages, 0, DOWNLOAD_LAYER, algs, 'Share Percentage', DOWNLOAD_LAYER)


def popularity_test(inter_arrival):
    np.random.seed(1)
    my_net = NetGenerator().get_g()
    ACCEPT_RATIO = "Accept Ratio"
    DOWNLOAD_LAYER = "Download (MB)"
    solvers = [
        NoShareSolver(my_net, 0),
        ShareSolver(my_net, 0),
        ShareSolver(my_net, 1),
        ShareSolver(my_net, 2),
        PopularitySolver(my_net, 1),
        PopularitySolver(my_net, 2)
    ]
    stats = {ACCEPT_RATIO: Stat.MEAN_MODE, DOWNLOAD_LAYER: Stat.MEAN_MODE}
    algs = [s.get_name() for s in solvers]
    stat_collector = StatCollector(algs, stats)
    #
    iterations = 5
    arrival_rate = 1.0 / inter_arrival
    layer_num = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    layer_sizes = [[300, 301], [150, 151], [100, 101], [75, 76], [60, 61]]
    layer_num_avg = []
    for i in range(len(layer_num)):
        np.random.seed(i * 100)
        Const.VNF_LAYER = layer_num[i]
        Const.LAYER_SIZE = layer_sizes[i]
        x = int((layer_num[i][1] - 1 + layer_num[i][0]) / 2.0)
        layer_num_avg.append(x)
        sfc_gen = SfcGenerator(my_net)
        run_name = "{}".format(x)
        print("run-name:", run_name)
        for itr in range(iterations):
            reqs = []
            req_num = 150
            t = 0
            for _ in range(req_num):
                reqs.append(sfc_gen.get_chain(t))
                t = t + int(np.ceil(np.random.exponential(1.0 / arrival_rate)))
            for solver in solvers:
                np.random.seed(itr * 1234)
                res, dl_vol = test(solver, reqs)
                stat_collector.add_stat(solver.get_name(), ACCEPT_RATIO, run_name, res)
                stat_collector.add_stat(solver.get_name(), DOWNLOAD_LAYER, run_name, dl_vol)

    machine_id = "ut"
    fig_test_id = "{}_popularity".format(machine_id)
    fig_2 = './result/{}_accept_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', layer_num_avg, 0, ACCEPT_RATIO, algs, 'No. of Layers', ACCEPT_RATIO)

    fig_2 = './result/{}_dl_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', layer_num_avg, 0, DOWNLOAD_LAYER, algs, 'No. of Layers', DOWNLOAD_LAYER)


def slack_time_test(inter_arrival):
    np.random.seed(1)
    my_net = NetGenerator().get_g()
    ACCEPT_RATIO = "Accept Ratio"
    DOWNLOAD_LAYER = "Download (MB)"
    solvers = [
        NoShareSolver(my_net, 0),
        ShareSolver(my_net, 0),
        ShareSolver(my_net, 6)
    ]
    stats = {ACCEPT_RATIO: Stat.MEAN_MODE, DOWNLOAD_LAYER: Stat.MEAN_MODE}
    algs = [s.get_name() for s in solvers]
    stat_collector = StatCollector(algs, stats)
    #
    iterations = 5
    arrival_rate = 1.0 / inter_arrival
    tau1s = [[1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10]]
    tau1_avg = []
    Const.VNF_LAYER = [3, 6]
    Const.LAYER_SIZE = [60, 101]
    sfc_gen = SfcGenerator(my_net)
    for i in range(len(tau1s)):
        np.random.seed(i * 100)
        Const.TAU1 = tau1s[i]
        x = int((tau1s[i][1] - 1 + tau1s[i][0]) / 2.0)
        tau1_avg.append(x)
        run_name = "{}".format(x)
        print("run-name:", run_name)
        for itr in range(iterations):
            reqs = []
            req_num = 150
            t = 0
            for _ in range(req_num):
                reqs.append(sfc_gen.get_chain(t))
                t = t + int(np.ceil(np.random.exponential(1.0 / arrival_rate)))
            for solver in solvers:
                np.random.seed(itr * 1234)
                res, dl_vol = test(solver, reqs)
                stat_collector.add_stat(solver.get_name(), ACCEPT_RATIO, run_name, res)
                stat_collector.add_stat(solver.get_name(), DOWNLOAD_LAYER, run_name, dl_vol)

    machine_id = "ut"
    fig_test_id = "{}_slack".format(machine_id)
    fig_2 = './result/{}_accept_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', tau1_avg, 0, ACCEPT_RATIO, algs, 'Avg. Slack', ACCEPT_RATIO)

    fig_2 = './result/{}_dl_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', tau1_avg, 0, DOWNLOAD_LAYER, algs, 'Avg. Slack', DOWNLOAD_LAYER)


def layer_num_test(inter_arrival):
    np.random.seed(1)
    my_net = NetGenerator().get_g()
    ACCEPT_RATIO = "Accept Ratio"
    DOWNLOAD_LAYER = "Download (MB)"
    solvers = [
        NoShareSolver(my_net, 0),
        ShareSolver(my_net, 0),
        ShareSolver(my_net, 6)
    ]
    stats = {ACCEPT_RATIO: Stat.MEAN_MODE, DOWNLOAD_LAYER: Stat.MEAN_MODE}
    algs = [s.get_name() for s in solvers]
    stat_collector = StatCollector(algs, stats)
    #
    iterations = 5
    arrival_rate = 1.0 / inter_arrival
    layer_num = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    layer_sizes = [[300, 301], [150, 151], [100, 101], [75, 76], [60, 61]]
    layer_num_avg = []
    for l in layer_num:
        layer_num_avg.append(int((l[1] - 1 + l[0]) / 2.0))
    for i in range(len(layer_num)):
        np.random.seed(i * 100)
        Const.VNF_LAYER = layer_num[i]
        Const.LAYER_SIZE = layer_sizes[i]
        x = int((layer_num[i][1] - 1 + layer_num[i][0]) / 2.0)
        sfc_gen = SfcGenerator(my_net)
        run_name = "{}".format(x)
        print("run-name:", run_name)
        for itr in range(iterations):
            reqs = []
            req_num = 150
            t = 0
            for _ in range(req_num):
                reqs.append(sfc_gen.get_chain(t))
                t = t + int(np.ceil(np.random.exponential(1.0 / arrival_rate)))
            for solver in solvers:
                np.random.seed(itr * 1234)
                res, dl_vol = test(solver, reqs)
                stat_collector.add_stat(solver.get_name(), ACCEPT_RATIO, run_name, res)
                stat_collector.add_stat(solver.get_name(), DOWNLOAD_LAYER, run_name, dl_vol)

    machine_id = "ut"
    fig_test_id = "{}_layer_num".format(machine_id)
    fig_2 = './result/{}_accept_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', layer_num_avg, 0, ACCEPT_RATIO, algs, 'No. of Layers', ACCEPT_RATIO)

    fig_2 = './result/{}_dl_ia{}'.format(fig_test_id, inter_arrival)
    stat_collector.write_to_file(fig_2 + '.txt', layer_num_avg, 0, DOWNLOAD_LAYER, algs, 'No. of Layers', DOWNLOAD_LAYER)


if __name__ == "__main__":
    my_argv = sys.argv[1:]
    test_type = "popularity"
    ia = 2
    opts, args = getopt.getopt(my_argv, "", ["inter-arrival=", "test-type="])
    for opt, arg in opts:
        if opt in ("--inter-arrival",):
            ia = int(arg)
        elif opt in ("--test-type",):
            print("test type is: {}".format(arg))
            test_type = arg
    if test_type == "slack":
        slack_time_test(ia)
    elif test_type == "layer":
        layer_num_test(ia)
    elif test_type == "popularity":
        popularity_test(ia)
    elif test_type == "share":
        share_percentage_test(ia)
    else:
        print("test: {} is not supported".format(test_type))
