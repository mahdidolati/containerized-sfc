import numpy as np
from ast import literal_eval


class QLearn:
    def __init__(self):
        self.q_vals = dict()
        self.alpha = 0.9
        self.gamma = 0.9
        self.epsilon = 1.0
        self.e_init = 1.0
        self.e_target = 0.1
        self.e_steps = 300

    def has_action(self, s):
        s_str = str(s)
        if s_str not in self.q_vals:
            return False
        if len(self.q_vals[s_str]) <= 0:
            return False
        if np.random.uniform(0, 1) <= self.epsilon:
            if self.epsilon > self.e_target:
                self.epsilon = self.epsilon - (self.e_init - self.e_target) / self.e_steps
            return False
        return True

    def get_action(self, s, a_lim=0):
        s_str = str(s)
        q_list = []
        q_sum = 0
        for a in self.q_vals[s_str]:
            will_remain = int(a[1:].split(",")[0])
            if will_remain == 0:
                return set()
            if will_remain < a_lim:
                q_list.append(a)
                q_sum = q_sum + self.q_vals[s_str][a]
        #
        if len(q_list) == 0:
            return None
        if q_sum == 0:
            selected_action = np.random.choice(a=q_list, size=1)
        else:
            prs = [self.q_vals[s_str][a]/q_sum for a in q_list]
            selected_action = np.random.choice(a=q_list, size=1, p=prs)
        set_str = "{" + selected_action[0][1:-1].split("{")[1]
        set_parsed = literal_eval(set_str)
        return set_parsed

    def add_transition(self, s1, a1, r1, s2):
        s1_str = str(s1)
        a1_str = str(a1)
        s2_str = str(s2)
        if s1_str != s2_str:
            if s1_str not in self.q_vals:
                self.q_vals[s1_str] = dict()
            if a1_str not in self.q_vals[s1_str]:
                self.q_vals[s1_str][a1_str] = 0.0
            if s2_str not in self.q_vals:
                self.q_vals[s2_str] = dict()
            s2_actions = [self.q_vals[s2_str][a2] - self.q_vals[s1_str][a1_str] for a2 in self.q_vals[s2_str]]
            if len(s2_actions) == 0:
                s2_actions.append(0)
            self.q_vals[s1_str][a1_str] = self.q_vals[s1_str][a1_str] + self.alpha * (
                r1 + self.gamma * max(s2_actions)
            )
        pass