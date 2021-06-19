from sfc import SfcGenerator
from net import NetGenerator
from solution import NoShareSolver, ShareSolver
import numpy as np
import scipy.stats
from constants import Const
from statistic_collector import StatCollector, Stat
import heapq


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
        ShareSolver(my_net, 0),
        ShareSolver(my_net, 5),
        ShareSolver(my_net, 10)
    ]
    stats = {ACCEPT_RATIO: Stat.MEAN_MODE}
    algs = [s.get_name() for s in solvers]
    stat_collector = StatCollector(algs, stats)
    #
    iterations = 5
    # layer_sizes = [[10, 590], [60, 540], [110, 490], [160, 440], [210, 390]]
    # layer_sizes = [[50, 150], [150, 250], [250, 350], [350, 450], [450, 550]]
    # 1, 5, 10, 15, 20
    layer_num = [[1, 2], [3, 8], [7, 14], [11, 20], [15, 26]]
    layer_num_avg = []
    for l in layer_num:
        layer_num_avg.append(int((l[1] - 1 + l[0]) / 2.0))
    for i in range(len(layer_num)):
        Const.VNF_LAYER = layer_num[i]
        x = int((layer_num[i][1] - 1 + layer_num[i][0]) / 2.0)
        # Const.LAYER_SIZE = [(300/x)-(120/x), (300/x)+(120/x)+1]
        sfc_gen = SfcGenerator(my_net)
        run_name = "{}".format(x)
        for itr in range(iterations):
            reqs = []
            req_num = 100
            for t in range(req_num):
                reqs.append(sfc_gen.get_chain(t))
            for solver in solvers:
                res = test(my_net, reqs)
                stat_collector.add_stat(solver.get_name(), ACCEPT_RATIO, run_name, res)

    fig_2 = './result/layer_num'
    stat_collector.write_to_file(fig_2 + '.txt', layer_num_avg, 0, ACCEPT_RATIO, algs,
                                 'No. of Switches', 'Success rate')


if __name__ == "__main__":
    main()