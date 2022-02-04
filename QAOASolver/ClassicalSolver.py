# %%
import numpy as np

# %%
def dynamic_programming(profits, weights, max_weight):
    R = np.zeros((len(profits) + 1, max_weight + 1))
    for i in range(1, len(profits) + 1):
        for j in range(1, max_weight + 1):
            if weights[i - 1] <= j:
                R[i][j] = max(profits[i - 1] + R[i - 1][j - weights[i - 1]], R[i - 1][j])
            else:
                R[i][j] = R[i - 1][j]
    return R[len(profits)][max_weight]

# %%
def dynamic_programming(items, max_weight):
    profits = [item.profit for item in items]
    weights = [item.weight for item in items]
    R = np.zeros((len(profits) + 1, max_weight + 1))
    for i in range(1, len(profits) + 1):
        for j in range(1, max_weight + 1):
            if weights[i - 1] <= j:
                R[i][j] = max(profits[i - 1] + R[i - 1][j - weights[i - 1]], R[i - 1][j])
            else:
                R[i][j] = R[i - 1][j]
    return R[len(profits)][max_weight]

# %%
def lazy_greedy(profits, weights, max_weight):
    items = [[i, profits[i], weights[i], profits[i] / weights[i], 0] for i in range(len(profits))]
    items.sort(key=(lambda x: x[3]), reverse=True)
    total_weight = 0
    r_stop = None
    for item in items:
        if total_weight + item[2] <= max_weight:
            total_weight += item[2]
            item[-1] = 1
        else:
            r_stop = item[3] if r_stop is None else r_stop
    items.sort(key=(lambda x: x[0]))
    return [i[-1] for i in items], r_stop

# %%


