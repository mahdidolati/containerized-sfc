from sfc import SfcGenerator
from net import NetGenerator
from solution import NoShareSolver, ShareSolver
from constants import Const
from statistic_collector import StatCollector, Stat
import heapq
import numpy as np


def test(solver, reqs):
    solver.my_net.reset()
    rate = 0.0
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
            if solver.solve(s, t, sampling_rate):
                rate = rate + 1
                heapq.heappush(events, (s.tau2+1, counter, "FINISH", s))
                counter += 1
        elif ev == "FINISH":
            solver.handle_sfc_eviction(s)
    return rate / len(reqs)


def main():
    my_net = NetGenerator().get_g()
    ACCEPT_RATIO = "Accept Ratio"
    solvers = [
        NoShareSolver(my_net, 0),
        ShareSolver(my_net, 0)
    ]
    stats = {ACCEPT_RATIO: Stat.MEAN_MODE}
    algs = [s.get_name() for s in solvers]
    stat_collector = StatCollector(algs, stats)
    #
    iterations = 5
    arrival_rate = 1.0 / 7.0
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
            req_num = 50
            t = 0
            for _ in range(req_num):
                reqs.append(sfc_gen.get_chain(t))
                t = t + int(np.ceil(np.random.exponential(1.0 / arrival_rate)))
            for solver in solvers:
                np.random.seed(itr * 1234)
                res = test(solver, reqs)
                stat_collector.add_stat(solver.get_name(), ACCEPT_RATIO, run_name, res)

    fig_2 = './result/layer_num'
    stat_collector.write_to_file(fig_2 + '.txt', layer_num_avg, 0, ACCEPT_RATIO, algs, 'No. of Layers', ACCEPT_RATIO)


if __name__ == "__main__":
    main()