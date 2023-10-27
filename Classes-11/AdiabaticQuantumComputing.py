import numpy as np
from numpy import kron
from numpy.linalg import eig
import matplotlib.pyplot as plt
# from pprint import pprint


# # Consider the list of integers
# marks = np.array([100, 39, 40, 78, 90, 32, 120])
# marks_ndex = np.argmin(marks)
# # Using index() method
# print("Minimum Index position: ", marks_ndex)

# --- CONSTANTS
h1 = 0.6
h2 = 0
h3 = 0
J12 = -1.1
J13 = -2.1
J23 = -3.8

# --- INITIAL MATRIX
mat_sz = np.array([[1, 0], [0, -1]])
mat_sx = np.array([[0, 1], [1, 0]])
mat_one = np.eye(2)
# sz_1 = kron(mat_sz, kron(mat_one, mat_one))

# pprint(sz_1)

# --- CREATE OPERATORS
operator_sx_1 = kron(mat_sx, kron(mat_one, mat_one))
operator_sx_2 = kron(mat_one, kron(mat_sx, mat_one))
operator_sx_3 = kron(mat_one, kron(mat_one, mat_sx))

operator_sz_1 = kron(mat_sz, kron(mat_one, mat_one))
operator_sz_2 = kron(mat_one, kron(mat_sz, mat_one))
operator_sz_3 = kron(mat_one, kron(mat_one, mat_sz))

oper_sz = [operator_sz_1, operator_sz_2, operator_sz_3]

# --- H0
hamiltonian_H0 = - operator_sx_1 - operator_sx_2 - operator_sx_3

# --- H1
hamiltonian_H1 = - J12 * np.dot(operator_sz_1, operator_sz_2) - J13 * np.dot(operator_sz_1, operator_sz_3) \
                 - J23 * np.dot(operator_sz_2, operator_sz_3) \
                 - h1 * operator_sz_1 - h2 * operator_sz_2 - h3 * operator_sz_3

# pprint(hamiltonian_H0)
# pprint(hamiltonian_H1)

# --- SET DIFFERENT LAMBDA AND CALCULATE
lambda_list = np.linspace(0.0, 1.0, num=1000)
energy_diff_list = []

for i in range(0, len(lambda_list)):
    val_lambda = lambda_list[i]
    hamiltonian = (1-val_lambda) * hamiltonian_H0 + val_lambda * hamiltonian_H1
    # find eigen energies and eigen vectors
    eigen_e, eigen_v = eig(hamiltonian)
    sorted_eigen_e = sorted(eigen_e)
    energy_difference = sorted_eigen_e[1] - sorted_eigen_e[0]
    # appending
    energy_diff_list.append(energy_difference)


fig, ax = plt.subplots(1, 1, figsize=(10.0, 8.0))
ax.set_title('Spacing energy', fontsize=18)
ax.plot(lambda_list, energy_diff_list)
# describe plot
ax.set_xlabel(r"$\lambda$", fontsize=18)
ax.set_ylabel(r"$E_{1} - E_{0}$", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True)
# ax.legend
plt.savefig(f'./Spacing-energy.png')
# plt.show()

# --- CALCULATE ADIABATIC EVOLUTION
T_aqc = 0
for i in range(0, len(lambda_list) - 1):
    delta_lambda = lambda_list[i+1] - lambda_list[i]
    delta_energy = energy_diff_list[i+1] - energy_diff_list[i]
    T_aqc = T_aqc + delta_lambda / delta_energy ** 2

print(r"optimal running time of the adiabatic evolution $T_{AQC}$:", T_aqc)

# --- CALCULATE THE EXPECTATION VALUE OF Sz_i
all_expectation_sz = []
for i in range(0, 3):
    expectation_sz_i = []
    for j in range(0, len(lambda_list)):
        val_lambda = lambda_list[j]
        hamiltonian = (1 - val_lambda) * hamiltonian_H0 + val_lambda * hamiltonian_H1
        # find eigen energies and eigen vectors
        eigen_e, eigen_v = eig(hamiltonian)
        # find the ground state
        ground_state_index = np.argmin(eigen_e)
        ground_vector = eigen_v[:, ground_state_index]

        # calculate expectation value
        expectation_val_sz_i = np.dot(np.conj(ground_vector), np.dot(oper_sz[i], ground_vector.T))
        # appending
        expectation_sz_i.append(expectation_val_sz_i)

    # appending
    all_expectation_sz.append(expectation_sz_i)

fig, ax = plt.subplots(1, 1, figsize=(10.0, 8.0))
ax.set_title('Expectation value of $S_{z}$', fontsize=18)
ax.plot(lambda_list, all_expectation_sz[0], label=r'$S^{z}_{1}$')
ax.plot(lambda_list, all_expectation_sz[1], label=r'$S^{z}_{2}$')
ax.plot(lambda_list, all_expectation_sz[2], label=r'$S^{z}_{3}$')

# describe plot
ax.set_xlabel(r"$\lambda$", fontsize=18)
ax.set_ylabel(r"$<\psi | S^{z}_{i} | \psi>$", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True)
# ax.legend
ax.legend()
plt.savefig(f'./Sz-expectation-value.png')
# plt.show()
