from HungarianBFS import *

import numpy as np

from scipy.optimize import linear_sum_assignment


def get_sum(g, result):
    s = 0.0
    for i in result:
        s += g[i[0], i[1]]
    return s


def is_equal(a, b):
    return np.abs(a-b) < 1e-8


def test():
    m = 100
    n = 10
    solve_for_max_cost = False


    g = np.random.random((m, n)) * 1000
    coefficient = 1

    hungarian = None


    if solve_for_max_cost:
        hungarian = MaxCostHungarian
        coefficient = -1
    else:
        hungarian = MinCostHungarian

    dfs = hungarian(g)
    dfs.solve()
    print(get_sum(g, dfs.get_matches()))

    row_ind, col_ind = linear_sum_assignment(coefficient * g)
    print(g[row_ind, col_ind].sum())
    b = is_equal(get_sum(g, dfs.get_matches()), g[row_ind, col_ind].sum())
    print(b)


test()
