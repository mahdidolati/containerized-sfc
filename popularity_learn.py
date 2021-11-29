import numpy as np
from ast import literal_eval
import heapq


class PLearn:
    def __init__(self):
        self.p_vals = dict()

    def get_action(self, removable, req_vol, layer_size):
        del_vol = 0
        to_del = set()
        h = list()
        cnt = 1
        for s in removable:
            s_p = 0
            if s in self.p_vals:
                s_p = self.p_vals[s]
            heapq.heappush(h, (s_p, cnt, s))
            cnt = cnt + 1
        while del_vol < req_vol:
            s_p, cnt, s = heapq.heappop(h)
            del_vol = del_vol + layer_size[s].size
            to_del.add(s)
        return to_del

    def add_inuse(self, inuse):
        for s in inuse:
            if s not in self.p_vals:
                self.p_vals[s] = 0
            self.p_vals[s] = self.p_vals[s] + 1