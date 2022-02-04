# %%
import numpy as np
import copy
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import BasicAer
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_state_qsphere
from scipy.optimize import minimize
import seaborn
import time    
from ClassicalSolver import lazy_greedy as lazy_greedy_imp
import pandas as pd


# %%
INIT_TYPES = ["constant", "lazy greedy", "smoothened", "nonbiased"]
SOLVING_TYPES = ["Copula", "Hourglass"]

# %%
class Subcircuit:

    def __init__(self, profits, weights, probs, thetas, gamma=None, beta=None):
        self.profits = profits
        self.weights = weights
        self.probs = probs
        self.thetas = thetas
        self.n_qubits = len(profits)
        self.gamma = gamma
        self.beta = beta
        self.qubits = QuantumRegister(self.n_qubits)
        self.circuit = QuantumCircuit(self.qubits)

    def get_circuit(self):
        return self.circuit
        

# %%
class InitializationGates(Subcircuit):

    def __init__(self, profits, weights, probs, thetas):
        super().__init__(profits, weights, probs, thetas)

        # Qubits werden in den rotierten Eigenzuständen gemäß den WSKs p_i initiiert
        for i in range(self.n_qubits):
            init = 2 * np.arccos(np.sqrt(1 - probs[i]))
            self.circuit.u(init, 0, 0, self.qubits[i])
        self.circuit.barrier()

    def get_circuit(self):
        return self.circuit

# %%
class CostGate(Subcircuit):

    def __init__(self, profits, weights, probs, thetas, gamma, beta):
        super().__init__(profits, weights, probs, thetas, gamma, beta)

        # auf jeden Qubit wird der unitäre Operator korrespondierend zu den Kosten des aktuellen Zustands angewandt
        for i in range(self.n_qubits):
            self.circuit.rz(self.gamma * profits[i] * 2, self.qubits[i])
        self.circuit.barrier()

# %%
class CopulaMixingGate(Subcircuit):
    
    def __init__(self, profits, weights, probs, thetas, gamma, beta):
        super().__init__(profits, weights, probs, thetas, gamma, beta)

        # Liste von interagierenden bzw. zu verschränkenden Qubits erstellen
        rng = list(range(self.n_qubits))
        indices = list(zip(rng[::2], rng[1::2]))
        rng.append(rng.pop(0))
        indices += list(zip(rng[::2], rng[1::2]))

        for (i, j) in indices:
            p1 = self.probs[i]
            p2 = self.probs[j]
            theta = self.thetas[i]

            # Wahrscheinlichkeiten bestimmen
            p21 = p2 + theta * p2 * (1 - p1) * (1 - p2)
            p2_1 = p2 - theta * p1 * p2 * (1 - p2)
            phi_p1 = 2 * np.arcsin(np.sqrt(p1))
            phi_p21 = 2 * np.arcsin(np.sqrt(p21))
            phi_p2_1 = 2 * np.arcsin(np.sqrt(p2_1))
            # Rotations-Circuit definieren
            q_s = QuantumRegister(2)
            R_p12 = QuantumCircuit(q_s)
            R_p12.ry(phi_p1, 0)
            # hier evtl cry-gate verwenden?
            R_p12.cu(phi_p21, 0, 0, 0, q_s[0], q_s[1])
            R_p12.x(0)
            # hier ebenso
            R_p12.cu(phi_p2_1, 0, 0, 0, q_s[0], q_s[1])
            R_p12.x(0)
            # Rotations-Circuits hintereinanderschalten
            self.circuit.barrier()
            self.circuit = self.circuit.compose(R_p12.inverse(), qubits=[i, j])
            self.circuit.rz(2 * self.beta, i)
            self.circuit.rz(2 * self.gamma, j)
            self.circuit = self.circuit.compose(R_p12, qubits=[i, j])
            self.circuit.barrier()

# %%
class HourglassMixingGate(Subcircuit):
    def __init__(self, profits, weights, probs, thetas, gamma, beta):
        super().__init__(profits, weights, probs, thetas, gamma, beta)

        for i in range(self.n_qubits):
            phi = 2 * np.arcsin(np.sqrt(self.probs[i]))
            self.circuit.ry(phi, self.qubits[i]).inverse()
            self.circuit.rz(2 * self.beta, self.qubits[i])
            self.circuit.ry(phi, self.qubits[i])

# %%
class TotalCircuit(object):

    # params: Parameter beta_i, gamma_i im Format[betas] + [gammas]
    def __init__(self, profits, weights, probs, thetas, params, type):
        t1 = time.time()
        self.profits = profits
        self.weights = weights
        self.probs = probs
        self.thetas = thetas
        self.p = len(params) // 2
        self.betas = params[:self.p]
        self.gammas = params[self.p:]
        self.n_qubits = len(profits)
        self.qubits = QuantumRegister(self.n_qubits)
        self.bits = ClassicalRegister(self.n_qubits)
        self.circuit = QuantumCircuit(self.qubits, self.bits)
        args = [profits, weights, probs, thetas]
        self.init_circuit = InitializationGates(*args)
        self.cost_gates = [CostGate(*args + [self.gammas[i], self.betas[i]]).circuit for i in range(len(self.gammas))]
        if type == "Copula":
            self.mixing_gates = [CopulaMixingGate(*args + [self.gammas[i], self.betas[i]]).circuit for i in range(len(self.gammas))]
        else:
            self.mixing_gates = [HourglassMixingGate(*args + [self.gammas[i], self.betas[i]]).circuit for i in range(len(self.gammas))]
        self.rng = list(range(self.n_qubits))
        
        # Initiierungsgates anwenden
        self.circuit = self.circuit.compose(self.init_circuit.get_circuit(), qubits=self.rng)

        # in p Durchgängen werden jeweils Cost- und Mixing-Unitary angewandt:
        for i in range(self.p):
            self.circuit = self.circuit.compose(self.cost_gates[i], qubits=self.rng)
            self.circuit = self.circuit.compose(self.mixing_gates[i], qubits=self.rng)
        t2 = time.time()

    def set_circuit(self, probs, thetas, params):
        circuit = self.circuit.copy()
        

# %%
class Preprocessor(object):


    def __init__(self, profits, weights, k):
        self.profits = profits
        self.weights, self.max_weight = weights
        self.n_qubits = len(self.profits)
        self.k = k

    def lazy_greedy(self):
        return lazy_greedy_imp(self.profits, self.weights, self.max_weight)

    def bitstring_to_probs(self, type):
        if type == "constant":
            return np.array([self.max_weight / sum(self.weights) for _ in range(self.n_qubits)])
        elif type == "lazy greedy":
            lg = self.lazy_greedy()
            r_stop = lg[1]
            return np.array((np.array(self.profits) / np.array(self.weights)) > r_stop).astype(int)
        elif type == "smoothened":
            lg = self.lazy_greedy()
            r_stop = lg[1]
            C = sum(self.weights) / self.max_weight - 1
            func = np.vectorize(lambda x: 1 / (1 + C * np.exp(-1 * self.k * (x - r_stop))))
            return func(np.array(self.profits) / np.array(self.weights))
        elif type == "nonbiased":
            return [0.5 for _ in range(self.n_qubits)]
        else:
            print("Invalider Initialisierungstyp")
            return

# %%
class Solver(object):

    def __init__(self, profits, weights, thetas, solving_type, init_type, k):
        self.profits = profits
        self.thetas = thetas
        self.weights, self.max_weight = weights
        self.circuit = None
        self.params_set = False
        self.solving_type = solving_type
        self.probs = Preprocessor(profits, weights, k).bitstring_to_probs(init_type)


    def set_params(self, params):
        t1 = time.time()
        self.circuit = TotalCircuit(self.profits, self.weights, self.probs, self.thetas, params, self.solving_type)
        self.params_set = True
        t2 = time.time()
        #print("Setup-time: " + str(t2 - t1))
        

    def process_outcome(self):
        t1 = time.time()
        if not self.params_set:
            print("Parameter nicht initialisiert")
            return 
        copy_circuit = self.circuit.circuit.copy()
        t2 = time.time()
        copy_circuit.measure_all(add_bits=False)
        simulator = AerSimulator(method="matrix_product_state")
        tcirc = transpile(copy_circuit, simulator)
        result = simulator.run(tcirc, shots=self.circuit.n_qubits).result()
        job = result.get_counts(0)
        # state = Statevector.from_int(0, 2 ** self.circuit.n_qubits)
        # state = state.evolve(copy_circuit)
        # job = state.sample_counts(max(10, self.circuit.n_qubits))
        t3 = time.time()
        #print("Runtime: " + str(t3 - t2))
        # alle gemessenen Bitstrings erhalten
        #results = list(job.result().get_counts(copy_circuit).keys())
        result_strings = list(job.keys())
        result_counts = list(job.values())
        # Stringformat in Int-array umwandeln
        result_df = pd.DataFrame(result_strings, result_counts).reset_index()

        # Resultierende Bitstrings in Werte ummünzen
        result_df.columns = ["Bitstrings", "Anzahl"]
        result_df["Profit"] = result_df["Bitstrings"].apply(lambda x: (np.array([int(c) for c in str(x)]) * np.array(self.profits)).sum())
        result_df["Gewicht"] = result_df["Bitstrings"].apply(lambda x: (np.array([int(c) for c in str(x)]) * np.array(self.weights)).sum())
        result_df["Profit"] = result_df["Profit"] * (result_df["Gewicht"] <= self.max_weight)
        # Herausfiltern derer Bitstrings, welche das erlaubte Gesamtgewicht überschreiten
        result_val_exp = (result_df["Profit"] * result_df["Anzahl"].astype(int)) / result_df["Anzahl"].astype(int).sum()
        result_val_best = result_df["Profit"].max()
        result_best_string = result_df[result_df["Profit"] == result_df["Profit"].max()]["Bitstrings"][0] if result_val_best > 0 else ""

        # Maximalen Wert mit zugehörigem Bitstring ausgeben
        t4 = time.time()
        #return max(results, key=(lambda x: x[1])) if len(results) > 0 else ([], 0)
        return result_val_exp, result_val_best, result_best_string

# %%
class GridSearcher(object):
    
    def __init__(self, items, max_weight, thetas, N_beta, N_gamma, p, solving_type, init_type, k=5):
        profits = [item.profit for item in items]
        weights = ([item.weight for item in items], max_weight)
        self.args = [profits, weights, thetas]
        self.N_beta = N_beta
        self.N_gamma = N_gamma
        self.p = p
        self.solver = Solver(*self.args + [solving_type, init_type, k])

        def generating_func(old_list, list, p):
            if p == 1:
                return old_list
            return [[e] + o_e for o_e in generating_func(old_list, list, p-1) for e in list]
        
        beta_starting_list = [np.pi * i / N_beta for i in range(N_beta)]
        self.betas = generating_func([[b] for b in beta_starting_list], beta_starting_list, self.p)
        gamma_starting_list = [2 * np.pi * i / N_gamma for i in range(N_gamma)]
        self.gammas = generating_func([[g] for g in gamma_starting_list], gamma_starting_list, self.p)
        self.res = None
        self.max = None

    def search_results(self):
        return self.search_results_for_p_1() if self.p == 1 else self.search_results_for_p_1()

    def search_results_for_p_greater_1(self):
        pass

    def search_results_for_p_1(self):
        all_params_tuples = [(self.betas[i] + self.gammas[j], (i, j)) for i in range(len(self.betas)) for j in range(len(self.gammas))]
        res = []
        i = 0
        for tpl in all_params_tuples:
            i += 1
            params = tpl[0]
            indices = tpl[1]
            self.solver.set_params(params)
            res.append((indices, self.solver.process_outcome()))
        self.res = res

        return res

    def get_max_result(self):
        if self.res is None:
            self.search_results()
        return max(self.res, key=(lambda x: x[1][1]))[1][1, 2]

    def get_max_string(self):
        max_val = self.get_max_result()
        return list(map(lambda x: x[1][0], list(filter(lambda x: x[1][1] == max_val, self.res))))[0]

    def visualize(self):
        array = np.zeros((self.N_beta, self.N_gamma))
        for o in self.res:
            ((i, j), (value, _, _)) = o
            array[i][j] = value
        ax =  seaborn.heatmap(array)
        return ax

# %%
class Item(object):

    def __init__(self, id, profit, weight):
        self.id = id
        self.profit = profit
        self.weight = weight

    def __str__(self) -> str:
        first_line = "Object {}:".format(self.id)
        second_line = "   - profit: {}".format(self.profit)
        third_line = "   - weight: {}".format(self.weight)
        return "\n".join([first_line, second_line, third_line])

# %%
n_qubits = 10
qubits = QuantumRegister(n_qubits)
bits = ClassicalRegister(n_qubits)
circuit = QuantumCircuit(qubits, bits)

# %%



