import numpy as np
from numpy import kron
from numpy.linalg import eig
import matplotlib.pyplot as plt
from pprint import pprint
import copy


# --------------------------------- NETWORK OF OUR NODES --------------------------------- #
# --- write the network of nodes, which we consider
N_nodes = 8  # number of nodes in our network
J_ij = -1

dic_Graph = {
    1: {"ngb_nodes": [2, 5, 6], "J_ij": [J_ij, J_ij, J_ij]},
    2: {"ngb_nodes": [3, 6, 7], "J_ij": [J_ij, J_ij, J_ij]},
    3: {"ngb_nodes": [4, 7, 8], "J_ij": [J_ij, J_ij, J_ij]},
    4: {"ngb_nodes": [8], "J_ij": [J_ij]},
    5: {"ngb_nodes": [6], "J_ij": [J_ij]},
    6: {"ngb_nodes": [7], "J_ij": [J_ij]},
    7: {"ngb_nodes": [8], "J_ij": [J_ij]}
}
# print(len(dic_Graph))
# print(dic_Graph[1]["J_ij"])

# --------------------------------- CREATING THE OPERATORS: Sz_{i} and Sx_{i} --------------------------------- #
# --- initial matrix
mat_sz = np.array([[1, 0], [0, -1]])
mat_sx = np.array([[0, 1], [1, 0]])
mat_one = np.eye(2)

# --- create operators
operators_sx = []
operators_sz = []

for i in range(0, N_nodes):
    matrix_order_sz = []
    matrix_order_sx = []

    # appending into `matrix_ordes`, prep to calculate Sx_{i}, Sy_{i}
    for j in range(0, N_nodes):
        if i == j:
            matrix_order_sz.append(mat_sz)
            matrix_order_sx.append(mat_sx)
        else:
            matrix_order_sz.append(mat_one)
            matrix_order_sx.append(mat_one)

    kron_matrix_sz = []
    kron_matrix_sx = []
    for j in range(N_nodes-1, 0, -1):
        if j == 7:
            # dealing with operator `Sz`
            kron_matrix_sz = kron(matrix_order_sz[j-1], matrix_order_sz[j])
            # dealing with operator `Sx`
            kron_matrix_sx = kron(matrix_order_sx[j-1], matrix_order_sx[j])
        else:
            # dealing with operator `Sz`
            kron_matrix_sz = kron(matrix_order_sz[j-1], kron_matrix_sz)
            # dealing with operator `Sx`
            kron_matrix_sx = kron(matrix_order_sx[j-1], kron_matrix_sx)

    # appending into list of operators
    operators_sz.append(kron_matrix_sz)
    operators_sx.append(kron_matrix_sx)

# --- do print to check: this is how we can check it
# pprint(operators_sz[-1])
# print()
# pprint(operators_sz[0])
#
# # --- just to check one more thing
# operator_sz_0 = kron(mat_one, kron(mat_one, kron(mat_one,
# kron(mat_one, kron(mat_one, kron(mat_one, kron(mat_one, mat_sz)))))))
# print()
# pprint(operator_sz_0)

# --------------------------------- UPDATING THE NODES --------------------------------- #
# --- initial values of
h_i = 0.9

dic_Node = {}
for i in range(0, N_nodes):
    dic_Node[i+1] = {"S_z": operators_sz[i], "S_x": operators_sx[i], "h_i": h_i}


# --------------------------------- FUNCTION TO COMPUTE THINGS --------------------------------- #
def find_hamiltonians(node_dict, graph_dict):
    """
    :param node_dict: dictionary contains: 'S_x', 'S_x', 'h_i'.
    :param graph_dict: dictionary contains: 'ngb_nodes', 'J_ij'.
    :return: Hamiltonian - matrix.
    """
    # --- calculate `H0 = − sum_{i} S_x_i`
    len_mat = len(node_dict[1]["S_x"])
    size_mat = (len_mat, len_mat)  # to resize our hamiltonians

    hamiltonian_H0 = np.zeros(size_mat)
    for index_i in range(1, len(node_dict) + 1):
        hamiltonian_H0 = hamiltonian_H0 + (- 1) * node_dict[index_i]["S_x"]

    # --- calculate H1 = - sum_{i<j} J_ij * S_z_i * S_z_j - sum_{i} h_i * S_z_i
    hamiltonian_H1 = np.zeros(size_mat)
    for index_i in range(1, len(node_dict) + 1):
        hamiltonian_H1 = hamiltonian_H1 + (- 1) * node_dict[index_i]["h_i"] * node_dict[index_i]["S_z"]

    for index_i in range(1, len(graph_dict) + 1):
        hamiltonian_H1_i = np.zeros(size_mat)
        for index_j in range(0, len(graph_dict[index_i]["ngb_nodes"])):
            val_J_ij = graph_dict[index_i]["J_ij"][index_j]
            S_z_i = node_dict[index_i]["S_z"]
            ngb_nodes = graph_dict[index_i]["ngb_nodes"][index_j]
            S_z_j = node_dict[ngb_nodes]["S_z"]
            # calculate hamiltonian
            hamiltonian_H1_i = hamiltonian_H1_i + (- 1) * val_J_ij * np.dot(S_z_i, S_z_j)

        # update main hamiltonian
        hamiltonian_H1 = hamiltonian_H1 + hamiltonian_H1_i

    return [hamiltonian_H0, hamiltonian_H1]

def delta_energy(_hamiltonian_H0, _hamiltonian_H1, _val_lambda):
    """
    :param _hamiltonian_H0: first part of hamiltonian.
    :param _hamiltonian_H1: second part of hamiltonian.
    :param _val_lambda: parameter of our Hamiltonian.
    :return: The spacing energy between the first expectation state and ground state of our hamiltonian.
        In another word: `delta_E = E_1(lambda) - E_0(lambda)`.
    """
    # find hamiltonian
    hamiltonian = (1 - _val_lambda) * _hamiltonian_H0 + _val_lambda * _hamiltonian_H1
    # find eigen energies and eigen vectors
    eigen_e, eigen_v = eig(hamiltonian)
    sorted_eigen_e = sorted(eigen_e)
    energy_difference = np.real(sorted_eigen_e[1]) - np.real(sorted_eigen_e[0])

    return energy_difference


# --------------------------------- TESTING DIFFERENCES MODELS OF INTERACTION --------------------------------- #
# this time different types of interation means the changing values of `J_ij` and `h_i`
# one thing is part of mistery If I have to change all of `J_ij` and `h_i` or I still
# have to hold the fact that for all nodes those parameters have the same vale.

def change_h_i(_node_dict, one_value=False):
    """
    :param one_value: just setting. Do you want same value for `h_i`? Or different for each component?
    :param _node_dict: dictionary contains: 'S_x', 'S_x', 'h_i'.
    :return: modify dictionary with changed the `h_i` values, which are contained in the dictionary.
    """
    # randomise
    if one_value is False:
        h_i_list = np.random.uniform(-1, 1, 8)
    else:
        h_i = np.random.uniform(-1, 1, 1)[0]
        h_i_list = [h_i] * 8
    # update
    for index in range(1, len(_node_dict) + 1):
        _node_dict[index]["h_i"] = h_i_list[index-1]

    return _node_dict

def change_J_ij(_graph_dict, one_value=False):
    """
    :param one_value: just setting. Do you want same value for `J_ij`? Or different for each component?
    :param _graph_dict: dictionary contains: 'ngb_nodes', 'J_ij'.
    :return: modify dictionary with changed the `J_ij` values, which are contained in the dictionary.
    """
    N_elms = len(_graph_dict)
    for index in range(1, N_elms + 1):
        ngb_nodes = len(_graph_dict[index]["ngb_nodes"])

        # randomise
        if one_value is False:
            J_ij_list = np.random.uniform(-1, 1, ngb_nodes)
        else:
            val_J_ij = np.random.uniform(-1, 1, 1)[0]
            J_ij_list = [val_J_ij] * ngb_nodes

        # update
        for ngb in range(0, ngb_nodes):
            _graph_dict[index]["J_ij"][ngb] = J_ij_list[ngb]

    return _graph_dict

# --- just to make sure that above function works properly
new_dic_Node = change_h_i(copy.deepcopy(dic_Node), one_value=False)
new_dic_Graph = change_J_ij(copy.deepcopy(dic_Graph), one_value=False)
print("Working of deepcopy")
for i in range(1, len(new_dic_Node) + 1):
    print("new h_i:", new_dic_Node[i]["h_i"], "old h_i:", dic_Node[i]["h_i"])

for i in range(1, len(new_dic_Graph) + 1):
    print("new J_ij:", new_dic_Graph[i]["J_ij"], "old J_ij:", dic_Graph[i]["J_ij"])

# --- fixed for all model
lambda_list = np.linspace(0.0, 1.0, num=1000)
energy_diff_init_list = []
N_models = 12  # how many models we will consider

# finding hamiltonians
ham_H0, ham_H1 = find_hamiltonians(dic_Node, dic_Graph)
# finding different energies
for i in range(0, len(lambda_list)):
    val_lambda = lambda_list[i]
    diff_energy = delta_energy(ham_H0, ham_H1, val_lambda)
    energy_diff_init_list.append(diff_energy)

# --------------------------------- COMPARISON: SAME J_ij, CHANGED FIXED h_i --------------------------------- #
models_diff_energy = [None] * N_models
diff_h_i = []

for model in range(0, N_models):
    energy_diff_list = []
    # change model
    new_dic_Node = change_h_i(copy.deepcopy(dic_Node), one_value=True)
    new_dic_Graph = copy.deepcopy(dic_Graph)

    # finding hamiltonians
    ham_H0, ham_H1 = find_hamiltonians(new_dic_Node, new_dic_Graph)
    # finding different energies
    for i in range(0, len(lambda_list)):
        val_lambda = lambda_list[i]
        diff_energy = delta_energy(ham_H0, ham_H1, val_lambda)
        energy_diff_list.append(diff_energy)

    # appending
    models_diff_energy[model] = energy_diff_list
    diff_h_i.append(new_dic_Node[1]["h_i"])

fig, ax = plt.subplots(1, 1, figsize=(10.0, 8.0))
ax.set_title(r'Spacing energy, different: $h_{i}$ (one fixed value)', fontsize=18)
# data
ax.plot(lambda_list, energy_diff_init_list, label=r'initial, $h_{i}$: ' + f'{dic_Node[1]["h_i"]}')
for i in range(0, N_models):
    energy_diff_list = models_diff_energy[i]
    ax.plot(lambda_list, energy_diff_list, label=r'new, $h_{i}$: ' + f'{"{:.4f}".format(diff_h_i[i])}')

# describe plot
ax.set_xlabel(r"$\lambda$", fontsize=18)
ax.set_ylabel(r"$E_{1} - E_{0}$", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)

# grid
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.4)
# ax.legend
ax.legend()
plt.savefig(f'./Spacing-energy-different-hi-one-fixed.png')
# plt.show()

# --------------------------------- COMPARISON: SAME h_i, CHANGED FIXED J_ij --------------------------------- #
models_diff_energy = [None] * N_models
diff_J_ij = []

for model in range(0, N_models):
    energy_diff_list = []
    # change model
    new_dic_Node = copy.deepcopy(dic_Node)
    new_dic_Graph = change_J_ij(copy.deepcopy(dic_Graph), one_value=True)

    # finding hamiltonians
    ham_H0, ham_H1 = find_hamiltonians(new_dic_Node, new_dic_Graph)
    # finding different energies
    for i in range(0, len(lambda_list)):
        val_lambda = lambda_list[i]
        diff_energy = delta_energy(ham_H0, ham_H1, val_lambda)
        energy_diff_list.append(diff_energy)

    # appending
    models_diff_energy[model] = energy_diff_list
    diff_J_ij.append(new_dic_Graph[1]["J_ij"][0])

fig, ax = plt.subplots(1, 1, figsize=(10.0, 8.0))
ax.set_title(r'Spacing energy, different: $J_{ij}$ (one fixed value)', fontsize=18)
# data
ax.plot(lambda_list, energy_diff_init_list, label=r'initial, $J_{ij}$: ' + f'{dic_Graph[1]["J_ij"][0]}')
for i in range(0, N_models):
    energy_diff_list = models_diff_energy[i]
    ax.plot(lambda_list, energy_diff_list, label=r'new, $J_{ij}$: ' + f'{"{:.4f}".format(diff_J_ij[i])}')

# describe plot
ax.set_xlabel(r"$\lambda$", fontsize=18)
ax.set_ylabel(r"$E_{1} - E_{0}$", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)

# grid
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.4)
# ax.legend
ax.legend()
plt.savefig(f'./Spacing-energy-different-Jij-one-fixed.png')
# plt.show()

# --------------------------------- COMPARISON: CHANGED FIXED J_ij, h_i --------------------------------- #
models_diff_energy = [None] * N_models
diff_J_ij = []
diff_h_i = []

for model in range(0, N_models):
    energy_diff_list = []
    # change model
    new_dic_Node = change_h_i(copy.deepcopy(dic_Node), one_value=True)
    new_dic_Graph = change_J_ij(copy.deepcopy(dic_Graph), one_value=True)

    # finding hamiltonians
    ham_H0, ham_H1 = find_hamiltonians(new_dic_Node, new_dic_Graph)
    # finding different energies
    for i in range(0, len(lambda_list)):
        val_lambda = lambda_list[i]
        diff_energy = delta_energy(ham_H0, ham_H1, val_lambda)
        energy_diff_list.append(diff_energy)

    # appending
    models_diff_energy[model] = energy_diff_list
    diff_J_ij.append(new_dic_Graph[1]["J_ij"][0])
    diff_h_i.append(new_dic_Node[1]["h_i"])

fig, ax = plt.subplots(1, 1, figsize=(10.0, 8.0))
ax.set_title(r'Spacing energy, different: $J_{ij}$, $h_{i}$ (one fixed value)', fontsize=18)
# data
ax.plot(lambda_list, energy_diff_init_list, label=r'initial, $h_{i}$: ' + f'{dic_Node[1]["h_i"]}, '
                                                  + r'$J_{ij}$: ' + f'{dic_Graph[1]["J_ij"][0]}')
for i in range(0, N_models):
    energy_diff_list = models_diff_energy[i]
    ax.plot(lambda_list, energy_diff_list, label=r'new, $h_{i}$: ' + f'{"{:.4f}".format(diff_h_i[i])}, '
                                                 + r'$J_{ij}$: ' + f'{"{:.4f}".format(diff_J_ij[i])}')

# describe plot
ax.set_xlabel(r"$\lambda$", fontsize=18)
ax.set_ylabel(r"$E_{1} - E_{0}$", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)

# grid
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.4)
# ax.legend
ax.legend()
plt.savefig(f'./Spacing-energy-different-Jij-and-hi-one-fixed.png')
# plt.show()
