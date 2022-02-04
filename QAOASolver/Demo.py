# %%
from TotalSolver import *
from TotalSolver import Item
from TotalSolver import GridSearcher

from ClassicalSolver import *
from ClassicalSolver import dynamic_programming

from Testing.InstanceGenerator.InstanceGenerator import create_instance

import numpy as np
import time

# %%
def main():
    profits, weights, max_weights = create_instance("strong_cor", 10, 1000)
    max_weight = max_weights[len(max_weights) // 2]
    items = [Item(i, profits[i], weights[i]) for i in range(len(profits))]
    print("Collecting item-data from Input.xslx...", end=" ")
    print("")
    time.sleep(4)
    print("Done")
    for item in items:
        print(item)
        time.sleep(1)
    quantum_solver = GridSearcher(items, max_weight, [-0.5 for _ in items], 10, 10, 1, "Copula", "smoothened")
    print("Calculating optimal solution: ...")
    quantum_sol_value = quantum_solver.get_max_result()
    quantum_sol_string = quantum_solver.get_max_string()
    classical_sol_value = dynamic_programming(items, max_weight)
    print("Done")
    print("Optimal quantum value: {} (classical: {})".format(str(quantum_sol_value), str(classical_sol_value)))
    print("Optimal item combinations: {}".format(quantum_sol_string))

# %%
if __name__ == "__main__":
    main()
# %% [markdown]
# 


