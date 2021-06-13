from sfc import SfcGenerator
from net import NetGenerator
from solution import solve
import numpy as np
import scipy.stats
from constants import Const


def test(my_net, reqs):
    my_net.reset()
    rate = 0.0
    sampling_rate = 1.0
    for s in reqs:
        if solve(my_net, s, s.arrival_time, sampling_rate):
            rate = rate + 1
    return rate / len(reqs)


def main():
    my_net = NetGenerator().get_g()
    #
    itrations = 10
    # layer_sizes = [[10, 590], [60, 540], [110, 490], [160, 440], [210, 390]]
    # layer_sizes = [[50, 150], [150, 250], [250, 350], [350, 450], [450, 550]]
    # 1, 5, 10, 15, 20
    layer_num = [[1, 2], [3, 8], [7, 14], [11, 20], [15, 26]]
    for i in range(len(layer_num)):
        Const.VNF_LAYER = layer_num[i]
        x = int((layer_num[i][1] - 1 + layer_num[i][0]) / 2.0)
        # Const.LAYER_SIZE = [(300/x)-(120/x), (300/x)+(120/x)+1]
        sfc_gen = SfcGenerator(my_net)
        a = []
        for itr in range(itrations):
            reqs = []
            req_num = 100
            for t in range(req_num):
                reqs.append(sfc_gen.get_chain(t))
            res = test(my_net, reqs)
            a.append(res)
        confidence = 0.95
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        h_l = h
        h_h = h
        if m-h < 0:
            h_l = m
        if m+h > 1:
            h_h = 1.0-m
        print(x, Const.LAYER_SIZE, m, h_l, h_h)


if __name__ == "__main__":
    main()