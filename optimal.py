import cvxpy as cp
import numpy as np
from mip_cvxpy import PYTHON_MIP


def get_last_t(reqs):
    t_max = 0
    for r in reqs:
        t_max = max(r.tau2, t_max)
    return int(t_max)


def solve_optimal(my_net, R, reqs):
    T = get_last_t(reqs)
    Lw, Lm = my_net.get_link_sets()
    B = my_net.get_all_base_stations()
    E = my_net.get_all_edge_nodes()
    a_var = cp.Variable((len(reqs), ), boolean=True)
    z_var = cp.Variable((len(E)*len(R)*T, ), boolean=True)
    y_var = cp.Variable((len(E)*len(R)*T, ), boolean=True)
    constraints = []
    # constraints += [z_var + y_var <= np.ones((len(E)*len(R), T))]
    constraints += [cp.sum(a_var[0:3]) <= 2]
    constraints += [cp.sum(a_var[3:16]) <= 1]

    objective = cp.Maximize(cp.sum(a_var))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CBC, verbose=1)

    print(a_var[0].value)

    return 0, 0