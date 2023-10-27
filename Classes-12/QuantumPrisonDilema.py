import numpy as np
import matplotlib.pyplot as plt
from numpy import kron
from pprint import pprint
from qiskit import QuantumRegister, ClassicalRegister, Aer, execute, QuantumCircuit
from scipy.linalg import expm
from qiskit.extensions import UnitaryGate


# --------------------- TASK: ONE --------------------- #
# --- qubit
q = QuantumRegister(2, 'qubit')  #
c = ClassicalRegister(2, 'bit')

# --- Circuit
circuit = QuantumCircuit(q, c)
circuit.h(q[0])
circuit.cnot(q[0], q[1])

circuit.measure(q, c)
print(circuit.draw())  # how to draw our circuit

# --- Simulator
backend = Aer.get_backend('qasm_simulator')  # set type of simulator
job = execute(circuit, backend, shots=100)
res = job.result()  # run simulation and get results
prob = res.get_counts(circuit)  # get counting
print("Tails Always Fails: '00' and Tails Never Fails: '11'. Our results:", prob)

val_00 = prob['00']
val_11 = prob['11']
x = ['|00>'] * val_00 + ['|11>'] * val_11
plt.hist(x)
plt.ylabel("counts")
plt.xlabel("state")
plt.title('Perfect Coin: Bonnie and Clyde')
plt.savefig(f'./Perfect-Coin.png')

# --------------------- TASK: TWO --------------------- #
q = QuantumRegister(2, 'qubit')  # two qubits: because we have two player
c = ClassicalRegister(2, 'bit')  # classical register each for each player

print("q:", q)
print("c:", c)
# --- constants
Num_shots = 220
gamma_list = np.linspace(0, np.pi/2, 180)
gamma_th1 = np.arcsin(np.sqrt(1/5))
gamma_th2 = np.arcsin(np.sqrt(2/5))
mat_D = np.array([[0, 1], [-1, 0]])
mat_Q = np.array([[1j, 0], [0, -1j]])

matrix_J = expm(- 1j * gamma_list[10] * kron(mat_D, mat_D) / 2)
matrix_J_dag = matrix_J.conj().T
# pprint(mat_J)

# --- set the payoffs
payoff_cc = 3
payoff_dc = 5
payoff_cd = 0
payoff_dd = 1
payoff_List = [payoff_cc, payoff_dc, payoff_cd, payoff_dd]

# ----------------------------- CREATE CIRCUIT AND CALCULATE PAYOFFS ----------------------------- #
# --- prepare experiment
def create_circuit_and_measurer(_matrix_Ua, _matrix_Ub, _gamma, print_mode=False):
    """
    :param print_mode: Do you want print how look circuit? Set: `True`.
    :param _matrix_Ua: set the strategy of the first player.
    :param _matrix_Ub: set the strategy of the second player.
    :param _gamma: constant to describe: dependent entanglement).
    :return: the results of our experiment.
    """
    mat_J = expm(- 1j * _gamma * kron(mat_D, mat_D) / 2)
    # mat_J = expm(- 1j * _gamma * kron(_matrix_Ua, _matrix_Ub) / 2)
    mat_J_dag = mat_J.conj().T

    # ---------------------- circuit ---------------------- #
    # circuit = QuantumCircuit(q, c)
    circuit = QuantumCircuit(q)

    gate = UnitaryGate(mat_J)
    circuit.append(gate, q)
    # --- adding two matrices which will differ
    mat_UA = _matrix_Ua
    mat_UB = _matrix_Ub
    circuit.unitary(mat_UA, q[0])
    circuit.unitary(mat_UB, q[1])

    # circuit.h(q[0])
    # circuit.h(q[1])

    # --- adding matrix J dagger
    gate = UnitaryGate(mat_J_dag)
    circuit.append(gate, q)

    # circuit.measure(q, c)
    if print_mode is True:
        print(circuit.draw())  # how to draw our circuit

    # ---------------------- Simulator ---------------------- #
    # backend = Aer.get_backend('qasm_simulator')
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuit, backend, shots=Num_shots)
    res = job.result()

    prob = res.get_counts(circuit)
    if print_mode is True:
        print(prob)
        print()

    return prob

# --- calculate the payoffs
def calculate_payoffs(payoff_list, prob_dict):
    """
    :param payoff_list: the list, which set the payoffs.
    :param prob_dict: the results of our experiment.
    :return: the payoff of fixed strategy.
    """
    # payoffs
    payoff_CC = payoff_list[0]
    payoff_DC = payoff_list[1]
    payoff_CD = payoff_list[2]
    payoff_DD = payoff_list[3]

    # results during game
    key_list = list(prob_dict.keys())
    payoff_of_game = 0

    for i in range(0, len(key_list)):
        results = key_list[i]

        if results == '00':
            payoff_of_game += prob_dict['00'] * payoff_CC
        elif results == '01':
            # remember that here we have "reversed" qubits
            payoff_of_game += prob_dict['01'] * payoff_DC
        elif results == '10':
            # remember that here we have "reversed" qubits
            payoff_of_game += prob_dict['10'] * payoff_CD
        else:
            payoff_of_game += prob_dict['11'] * payoff_DD

    return payoff_of_game

def calculate_payoffs_vol2(payoff_list, prob_dict, amplitude=Num_shots):
    """
    :param amplitude:
    :param payoff_list: the list, which set the payoffs.
    :param prob_dict: the results of our experiment.
    :return: the payoff of fixed strategy.
    """
    # payoffs
    payoff_CC = payoff_list[0]
    payoff_DC = payoff_list[1]
    payoff_CD = payoff_list[2]
    payoff_DD = payoff_list[3]

    # results during game
    key_list = list(prob_dict.keys())
    payoff_of_game = 0

    for i in range(0, len(key_list)):
        results = key_list[i]

        if results == '00':
            payoff_of_game += prob_dict['00'] * payoff_CC / Num_shots
        elif results == '01':
            # remember that here we have "reversed" qubits
            payoff_of_game += prob_dict['01'] * payoff_DC / Num_shots
        elif results == '10':
            # remember that here we have "reversed" qubits
            payoff_of_game += prob_dict['10'] * payoff_CD / Num_shots
        else:
            payoff_of_game += prob_dict['11'] * payoff_DD / Num_shots

    return payoff_of_game

# ----------------------------- CALCULATE THE RESULTS ----------------------------- #
strategy_Q_Q = create_circuit_and_measurer(mat_Q, mat_Q, gamma_list[2], print_mode=True)

# ----------------------------- CALCULATE THE RESULTS ----------------------------- #
CC_list = []
DC_list = []
CD_list = []
DD_List = []
for i in range(0, len(gamma_list)):
    gamma_val = gamma_list[i]
    # choice of strategy
    strategy_Q_Q = create_circuit_and_measurer(mat_Q, mat_Q, gamma_val, print_mode=False)
    strategy_D_Q = create_circuit_and_measurer(mat_D, mat_Q, gamma_val, print_mode=False)
    strategy_Q_D = create_circuit_and_measurer(mat_Q, mat_D, gamma_val, print_mode=False)
    strategy_D_D = create_circuit_and_measurer(mat_D, mat_D, gamma_val, print_mode=False)
    # calculate the payoffs
    CC_payoff = calculate_payoffs(payoff_List, strategy_Q_Q)
    DC_payoff = calculate_payoffs(payoff_List, strategy_D_Q)
    CD_payoff = calculate_payoffs(payoff_List, strategy_Q_D)
    DD_payoff = calculate_payoffs(payoff_List, strategy_D_D)

    # CC_payoff = calculate_payoffs_vol2(payoff_List, strategy_Q_Q)
    # DC_payoff = calculate_payoffs_vol2(payoff_List, strategy_D_Q)
    # CD_payoff = calculate_payoffs_vol2(payoff_List, strategy_Q_D)
    # DD_payoff = calculate_payoffs_vol2(payoff_List, strategy_D_D)
    # appending
    CC_list.append(CC_payoff)
    DC_list.append(DC_payoff)
    CD_list.append(CD_payoff)
    DD_List.append(DD_payoff)

# ----------------------------- PLOT THE RESULTS ----------------------------- #
fig, ax = plt.subplots(1, 1, figsize=(12.0, 8.0))
ax.set_title('Payoffs depending on fixed strategy', fontsize=18)

# --- strategy
ax.plot(gamma_list, CC_list, label=r'strategy: CC')
ax.plot(gamma_list, DC_list, label=r'strategy: DC')
ax.plot(gamma_list, CD_list, label=r'strategy: CD')
ax.plot(gamma_list, DD_List, label=r'strategy: DD')
# --- lines
plt.axvline(x=gamma_th1, color='black', ls='--', label=r'$\arcsin{\sqrt{1/5}}$')
plt.axvline(x=gamma_th2, color='grey', ls='--', label=r'$\arcsin{\sqrt{2/5}}$')

# describe plot
ax.set_xlabel(r"$\gamma$", fontsize=18)
ax.set_ylabel(r"payoff", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True)
# ax.legend
ax.legend()
plt.savefig(f'./Payoffs-depending-on-strategy-statevector_simulator.png')
# plt.savefig(f'./Payoffs-depending-on-strategy-qasm_simulator.png')
# plt.show()
