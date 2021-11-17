import numpy as np


class QLearn:
    def __init__(self):
        self.q_vals = dict()
        self.alpha = 0.1
        self.gamma = 0.1

    def has_action(self, s):
        return s in self.q_vals

    def get_action(self, s):
        q_sum = sum([self.q_vals[s][a] for a in self.q_vals[s]])
        prs = [self.q_vals[s][a]/q_sum for a in self.q_vals[s]]
        selected_action = np.random.choice(a=self.q_vals[s], size=1, p=prs)
        return selected_action

    def add_transition(self, s1, a1, r1, s2):
        self.q_vals[s1][a1] = self.q_vals[s1][a1] + self.alpha * (
            r1 + self.gamma * max([self.q_vals[s2][a2] - self.q_vals[s1][a1] for a2 in self.q_vals[s2]])
        )