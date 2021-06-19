import numpy as np


def usable_node(my_net, s, c, chain_req, i, t, delay_budget):
    if my_net.g.nodes[c]["nd"].cpu_avail(t) < chain_req.cpu_req(i):
        return False
    if my_net.g.nodes[c]["nd"].ram_avail(t) < chain_req.ram_req(i):
        return False
    R, d = my_net.get_missing_layers(c, chain_req, i, chain_req.tau1)
    if my_net.g.nodes[c]["nd"].disk_avail() < d:
        return False
    dl_result, dl_obj = my_net.do_layer_dl_test(c, d, t, chain_req.tau1-1)
    if not dl_result:
        return False
    if s != c:
        path_bw, path_delay, links = my_net.get_biggest_path(s, c, t, delay_budget)
        dl_obj.cancel_download()
        if path_bw < chain_req.vnf_in_rate(i):
            return False
    else:
        dl_obj.cancel_download()
    return True


def solve(my_net, chain_req, t, sr):
    prev = chain_req.entry_point
    delay_budge = chain_req.max_delay
    active_dls = []
    for i in range(len(chain_req.vnfs)):
        cur_budge = delay_budge / (len(chain_req.vnfs) - i)
        N1 = my_net.get_random_edge_nodes(sr)
        N1 = np.append(N1, "c")
        C = list()
        for c in N1:
            if usable_node(my_net, prev, c, chain_req, i, t, cur_budge):
                C.append(c)
        if len(C) == 0:
            for a in active_dls:
                a.cancel_download()
            return False
        m = np.random.choice(C)
        my_net.g.nodes[m]["nd"].embed(chain_req, i)
        if prev != m:
            _, d = my_net.get_missing_layers(m, chain_req, i, chain_req.tau1)
            _, dl_obj = my_net.do_layer_dl_test(m, d, t, chain_req.tau1 - 1)
            active_dls.append(dl_obj)
            path_bw, path_delay, links = my_net.get_biggest_path(prev, m, t, cur_budge)
            delay_budge = delay_budge - path_delay
            for l in links:
                l.embed(chain_req, i)
        prev = m
    return True
