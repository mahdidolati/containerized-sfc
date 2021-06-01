import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations


class NetGenerator:
    def __init__(self):
        base_station_loc = [(0, 6), (3, 6), (6, 6), (0, 3), (0, 0), (3, 0), (6, 0)]
        cloud_loc = (10, 3)
        self.g = nx.MultiGraph()
        for n in range(len(base_station_loc)):
            self.g.add_node("b{}".format(n), type="base-station", loc=base_station_loc[n])
        self.edge_num = 15
        for n in range(self.edge_num):
            self.g.add_node("e{}".format(n), type="edge", loc=np.random.uniform(0.5, 5.5, 2))
        self.g.add_node("c", type="cloud", loc=cloud_loc)
        #
        for n in range(len(base_station_loc)):
            c = self.get_closest("b{}".format(n))
            self.g.add_edge("b{}".format(n), "e{}".format(c), type="wired")
        c = self.get_closest("c")
        self.g.add_edge("c", "e{}".format(c), type="wired")
        for l in combinations(range(self.edge_num), 2):
            e1 = "e{}".format(l[0])
            e2 = "e{}".format(l[1])
            if np.random.uniform(0, 1.0) < 0.1:
                self.g.add_edge(e1, e2, type="wired")
            if np.linalg.norm(self.g.nodes[e1]["loc"] - self.g.nodes[e2]["loc"]) < 2.0:
                self.g.add_edge(e1, e2, type="mmWave")

    def get_g(self):
        fig, ax = plt.subplots()
        x = []
        y = []
        for n in self.g.nodes():
            x.append(self.g.nodes[n]["loc"][0])
            y.append(self.g.nodes[n]["loc"][1])
        ax.plot(x, y, '.b')
        for n in self.g.nodes():
            ax.annotate(n, self.g.nodes[n]["loc"])
        for e in self.g.edges(data=True):
            for j in self.g[e[0]][e[1]]:
                line_t = 'r-'
                if self.g[e[0]][e[1]][j]["type"] == "mmWave":
                    line_t = 'b-'
                ax.plot([self.g.nodes[e[0]]["loc"][0], self.g.nodes[e[1]]["loc"][0]],
                        [self.g.nodes[e[0]]["loc"][1], self.g.nodes[e[1]]["loc"][1]], line_t)
        plt.show()
        return self.g.edges()

    def get_closest(self, b):
        ds = np.infty
        closest = None
        l1 = self.g.nodes[b]["loc"]
        for n in range(self.edge_num):
            l2 = self.g.nodes["e{}".format(n)]["loc"]
            d = np.linalg.norm(l1 - l2)
            if closest is None or d < ds:
                ds = d
                closest = n
        return closest