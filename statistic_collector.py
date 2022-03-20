#!/usr/bin/python
# RCCO, (C) Mahdi Dolati. License: AGPLv3
import numpy as np


class Stat:
    LIST_MODE = 'list'
    MEAN_MODE = 'mean'
    MEDIAN_MODE = 'median'
    SUM_MODE = 'sum'
    SQRT_MEAN_SQUARE_MODE = 'sqrt-mean-square'
    LEN_MODE = 'len'
    T95_MODE = 't-test-95'
    MAX_MODE = 'max'

    def __init__(self, mode):
        self.mode = mode
        self.stat = []

    def add_stat(self, s):
        self.stat.append(s)

    def get_list(self):
        return self.stat

    def get_stat(self):
        if self.mode == Stat.LIST_MODE:
            return self.stat
        elif self.mode == Stat.MEAN_MODE:
            return np.mean(self.stat)
        elif self.mode == Stat.MEDIAN_MODE:
            return np.median(self.stat)
        elif self.mode == Stat.SUM_MODE:
            return np.sum(self.stat)
        elif self.mode == Stat.SQRT_MEAN_SQUARE_MODE:
            return np.sqrt(np.mean(np.square(self.stat)))
        elif self.mode == Stat.LEN_MODE:
            return len(self.stat)
        elif self.mode == Stat.T95_MODE:
            return self.calc_t95()
        elif self.mode == Stat.MAX_MODE:
            return np.max(self.stat)
        else:
            print('stat not defined!')
        return None

    def calc_t95(self):
        if len(self.stat) <= 1:
            return (100, 100)
        t_95 = [12.71, 4.30, 3.18, 2.77, 2.57, 2.44, 2.36, 2.30,
                2.26, 2.22, 2.20, 2.17, 2.16, 2.14, 2.13, 2.12,
                2.11, 2.10, 2.09, 2.08]
        if len(self.stat) <= 1:
            t_95n = t_95[0]
        elif len(self.stat) >= 22:
            t_95n = t_95[-1]
        else:
            t_95n = t_95[len(self.stat) - 2]
        min_stat = min(self.stat)
        max_stat = max(self.stat)
        mean_stat = np.mean(self.stat)
        std1 = np.sqrt(np.sum(np.square(self.stat - mean_stat))) / (len(self.stat) - 1)
        y1_err = t_95n * std1 / np.sqrt(len(self.stat))
        return min(y1_err, abs(mean_stat-min_stat)), min(y1_err, abs(mean_stat+max_stat))

class StatCollector:
    def __init__(self, algs, stats, dilim='-'):
        self.algs = algs
        self.stats = stats
        self.stat_set = {}
        self.delim = dilim

    def add_stat(self, alg_name, stat_name, run_name, val):
        if run_name not in self.stat_set:
            self.stat_set[run_name] = {}
            for alg in self.algs:
                self.stat_set[run_name][alg] = {}
                for stat in self.stats:
                    self.stat_set[run_name][alg][stat] = Stat(self.stats[stat])

        self.stat_set[run_name][alg_name][stat_name].add_stat(val)

    def make_accumulated(self, stat_name, x_index):
        sorted_run_names = []
        for r in self.stat_set:
            if len(sorted_run_names) == 0:
                sorted_run_names.append(r)
            else:
                cur_i = 0
                while cur_i < len(sorted_run_names) and float(sorted_run_names[cur_i].split(self.delim)[x_index]) < float(r.split(self.delim)[x_index]):
                    cur_i += 1
                sorted_run_names.insert(cur_i, r)
        #
        for i in range(1, len(sorted_run_names)):
            r = sorted_run_names[i]
            r_1 = sorted_run_names[i-1]
            for a in self.algs:
                self.stat_set[r][a][stat_name].stat = self.stat_set[r][a][stat_name].stat + self.stat_set[r_1][a][stat_name].stat
        #
        for a in self.algs:
            sum_last = sum(self.stat_set[sorted_run_names[-1]][a][stat_name].stat)
            if sum_last == 0:
                continue
            for r in self.stat_set:
                self.stat_set[r][a][stat_name].stat = [1.0 * v / sum_last for v in self.stat_set[r][a][stat_name].stat]

    def print_stat(self):
        for run_name in self.stat_set:
            print(run_name)
            for stat in self.stats:
                for alg in self.algs:
                    print('\t\t', alg, stat, self.stat_set[run_name][alg][stat].get_stat())

    def write_to_file(self, filename, x, x_index, stat_name, algs, xtitle, ytitle):
        w = open(filename, 'w')
        x_idx = dict(zip(x, range(len(x))))
        alg_idx = dict(zip(algs, range(len(algs))))
        curve_num = len(algs)
        ys = []
        for _ in range(curve_num):
            y = [0.0] * len(x)
            ys.append(y)
        ys_err = []
        for _ in range(curve_num):
            y_l = [0.0] * len(x)
            y_h = [0.0] * len(x)
            y = [y_l, y_h]
            ys_err.append(y)
        for run_name in self.stat_set:
            x_tick = float(run_name.split(self.delim)[x_index])
            for alg in algs:
                ys[alg_idx[alg]][x_idx[x_tick]] = self.stat_set[run_name][alg][stat_name].get_stat()
                ys_err_low, ys_err_high = self.stat_set[run_name][alg][stat_name].calc_t95()
                ys_err[alg_idx[alg]][0][x_idx[x_tick]] = ys_err_low
                ys_err[alg_idx[alg]][1][x_idx[x_tick]] = ys_err_high
        w.write('ys_err;%s\n' % str(ys_err))
        w.write('x;%s\n' % str(x))
        w.write('ys;%s\n' % str(ys))
        w.write('ls;%s\n' % str(algs))
        w.write('xtitle;%s\n' %xtitle)
        w.write('ytitle;%s\n' %ytitle)
        w.close()

    def write_to_file_sub_catagory(self, filename, x, x_index, c, c_index, stat_name, algs, xtitle, ytitle):
        w = open(filename, 'w')
        x_idx = dict(zip(x, range(len(x))))
        ls = []
        for alg in algs:
            for c_i in c:
                ls.append('%s %s' %(alg, float(c_i)))
        alg_idx = dict(zip(ls, range(len(ls))))
        curve_num = len(ls)
        ys = []
        for _ in range(curve_num):
            y = [0.0] * len(x)
            ys.append(y)
        ys_err = []
        for _ in range(curve_num):
            y_l = [0.0] * len(x)
            y_h = [0.0] * len(x)
            y = [y_l, y_h]
            ys_err.append(y)
        for run_name in self.stat_set:
            x_tick = float(run_name.split(self.delim)[x_index])
            for alg in algs:
                c_i = float(run_name.split(self.delim)[c_index])
                ls_i = '%s %s' %(alg, c_i)
                ys[alg_idx[ls_i]][x_idx[x_tick]] = self.stat_set[run_name][alg][stat_name].get_stat()
                ys_err_low, ys_err_high = self.stat_set[run_name][alg][stat_name].calc_t95()
                ys_err[alg_idx[ls_i]][0][x_idx[x_tick]] = ys_err_low
                ys_err[alg_idx[ls_i]][1][x_idx[x_tick]] = ys_err_high
        w.write('ys_err;%s\n' % str(ys_err))
        w.write('x;%s\n' % str(x))
        w.write('ys;%s\n' % str(ys))
        w.write('ls;%s\n' % str(ls))
        w.write('xtitle;%s\n' % xtitle)
        w.write('ytitle;%s\n' % ytitle)
        w.close()

    def write_to_file_alg_x(self, filename, c, c_index, stat_name, algs, xtitle, ytitle):
        w = open(filename, 'w')
        ls = c
        alg_idx = dict(zip(algs, range(len(algs))))
        c_idx = dict(zip(c, range(len(c))))
        ys = []
        for _ in range(len(c)):
            y = [0.0] * len(algs)
            ys.append(y)
        ys_err = []
        for _ in range(len(c)):
            y_l = [0.0] * len(algs)
            y_h = [0.0] * len(algs)
            y = [y_l, y_h]
            ys_err.append(y)
        for run_name in self.stat_set:
            c_tick = float(run_name.split(self.delim)[c_index])
            c_tick_idx = c_idx[c_tick]
            for alg in algs:
                ys[c_tick_idx][alg_idx[alg]] = self.stat_set[run_name][alg][stat_name].get_stat()
                ys_err_low, ys_err_high = self.stat_set[run_name][alg][stat_name].calc_t95()
                ys_err[c_tick_idx][0][alg_idx[alg]] = ys_err_low
                ys_err[c_tick_idx][1][alg_idx[alg]] = ys_err_high
        w.write('ys_err;%s\n' % str(ys_err))
        w.write('x;%s\n' % str(algs))
        w.write('ys;%s\n' % str(ys))
        w.write('ls;%s\n' % str(ls))
        w.write('xtitle;%s\n' % xtitle)
        w.write('ytitle;%s\n' % ytitle)
        w.close()

    def write_to_file_no_x(self, filename, stat_name, algs, xtitle, ytitle):
        w = open(filename, 'w')
        x_idx = dict(zip(algs, range(len(algs))))
        curve_num = len(algs)
        ys = []
        for _ in range(curve_num):
            y = []
            ys.append(y)

        for run_name in self.stat_set:
            for alg in algs:
                ys[x_idx[alg]] = ys[x_idx[alg]] + self.stat_set[run_name][alg][stat_name].stat

        w.write('x;%s\n' % str(algs))
        w.write('ys;%s\n' % str(ys))
        w.write('ls;%s\n' % str(algs))
        w.write('xtitle;%s\n' % xtitle)
        w.write('ytitle;%s\n' % ytitle)
        w.close()