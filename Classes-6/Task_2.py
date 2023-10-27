import numpy as np
import os
# from pprint import pprint
import matplotlib.pyplot as plt
import random

# --------------- HELPFUL FUNCTION --------------- #
def make_directory(_path_name):
    """ Create directory with fixed name"""
    try:
        # Create target Directory
        os.mkdir(_path_name)
        print("Directory ", _path_name,  " Created ")
    except FileExistsError:
        print("Directory ", _path_name,  " already exists")

# --------------- Training matrices --------------- #
mat_zero = np.matrix([
    -1, 1, 1, 1, -1,
    1, -1, -1, -1, 1,
    1, -1, -1, -1, 1,
    1, -1, -1, -1, 1,
    1, -1, -1, -1, 1,
    -1, 1, 1, 1, -1
])
# mat_zero_copy = mat_zero.copy()
# mat_zero_copy[0, 0] = 1
# print(mat_zero)
# print(mat_zero_copy)

mat_one = np.matrix([
    -1, 1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1
])

mat_two = np.matrix([
    1, 1, 1, -1, -1,
    -1, -1, -1, 1, -1,
    -1, -1, -1, 1, -1,
    -1, 1, 1, -1, -1,
    1, -1, -1, -1, -1,
    1, 1, 1, 1, 1,
])

training_mat = [mat_zero, mat_one, mat_two]

# --------------- Input matrices: presents how training looks --------------- #
mat_noisy0 = np.matrix([
    -1, 1, 1, 1, -1,
    1, -1, -1, -1, -1,
    1, -1, -1, -1, 1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, 1, -1, -1,
])

mat_noisy2 = np.matrix([
    1, 1, 1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, 1, -1, -1,
    1, -1, -1, -1, -1,
    1, 1, -1, -1, 1,
])

mat_noisy2b = np.matrix([
    1, 1, 1, -1, -1,
    -1, -1, -1, 1, -1,
    -1, -1, -1, 1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
])

test_mats = [mat_noisy0, mat_noisy2, mat_noisy2b]
test_mat_names = ["noisy0", "noisy2", "noisy2b"]

# --------------- CRATE TRAINING MODEL --------------- #
def hopfield_network(training_matrices):
    """
    :param training_matrices: list of matrices, which we will use to training our network.
    :return: matrix, which describes network `after training`.
    """
    n_matrix = len(training_matrices)
    size_matrix = len(training_matrices[0].A[0])

    w_matrix = np.zeros((size_matrix, size_matrix))  # network after training
    for i in range(0, n_matrix):
        # create training pattern
        matrix = training_matrices[i]
        pattern = matrix.flatten()
        # update network
        w_matrix = w_matrix + np.outer(pattern, pattern) / n_matrix

    # Subtract identity matrix
    identity_matrix = np.identity(size_matrix)
    w_matrix = w_matrix - identity_matrix

    return np.matrix(w_matrix)

def test_network_asynchronously(pattern, w_matrix, steps=10):
    """
    :param pattern: pattern, which we will use as starting configuration
        of network.
    :param w_matrix: pattern, which we get after training.
    :param steps: number of updates, which we want make.
    :return: ending configuration and list contain energy of whole network after
        updating the network.
    """
    n_elements = pattern.size
    E_network_list = []
    for step in range(0, steps):
        # do one-step operation of hopfield network
        E_list = []
        # ordering of choosing spin, which are picked up randomly
        order_index = random.sample(range(n_elements), n_elements)

        for i in range(0, n_elements):
            index = order_index[i]
            x_i = pattern[0, index]
            # flipped spin
            if x_i == -1:
                x_i_flipped = 1
            else:
                x_i_flipped = -1
            pattern_flipped = pattern.copy()
            pattern_flipped[0, index] = x_i_flipped
            # calculate: sum over j of `w_ij * x_j`
            h_i = w_matrix[index].dot(pattern.T)
            h_i_flipped = w_matrix[index].dot(pattern_flipped.T)
            # compute energy
            E_i = float(- 0.5 * x_i * h_i)
            E_i_flipped = float(- 0.5 * x_i_flipped * h_i_flipped)
            if E_i_flipped < E_i:
                # we flip spin
                E_list.append(E_i_flipped)
                pattern = pattern_flipped
            else:
                E_list.append(E_i)
        # update test matrix
        pattern = np.sign(w_matrix.dot(pattern.T).T)
        E_network = sum(E_list)
        E_network_list.append(E_network)

    return pattern, E_network_list

# --------------- CHECK THE RESULTS --------------- #
if __name__ == '__main__':
    directory_path_name = "./Task2"
    make_directory(directory_path_name)
    w_mat = hopfield_network(training_mat)  # results of our training
    n_test = len(test_mats)
    Energy_network_list = []
    E_all = []

    for j in range(0, n_test):
        # control name
        name = test_mat_names[j]

        # --- PLOT: starting-configuration
        pattern_plot = np.asarray(test_mats[j]).reshape((6, 5))  # reshape pattern

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(pattern_plot, interpolation='nearest', cmap='viridis')
        cax.set_clim(vmin=0, vmax=1)
        cbar = fig.colorbar(cax, ticks=[0, 0.3, 0.5, 1], orientation='vertical')
        # plt.show()
        plt.savefig(directory_path_name + '/starting-configuration-' + name + '.png', dpi=300)
        plt.close(fig)

        # --- do evolution of pattern configuration
        test_mat, Energy_network_list = test_network_asynchronously(test_mats[j], w_mat)
        steps_list = np.arange(0, len(Energy_network_list), 1)
        E_all.append(Energy_network_list)

        # --- PLOT: ending configuration pattern
        pattern_plot = np.asarray(test_mat).reshape((6, 5))  # reshape pattern

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(pattern_plot, interpolation='nearest', cmap='viridis')
        cax.set_clim(vmin=0, vmax=1)
        cbar = fig.colorbar(cax, ticks=[0, 0.3, 0.5, 1], orientation='vertical')
        # plt.show()
        plt.savefig(directory_path_name + '/' + name + '.png', dpi=300)
        plt.close(fig)

        # --- PLOT: energy of network during updates
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        ax.set_title(f'Energy of network after updates', fontsize=22, loc='left')
        # data
        ax.plot(steps_list, Energy_network_list, '-o', color='tab:brown', label=f"Energy of network: {name}")
        # description
        ax.set_ylabel(r"$E_{network}$", fontsize=18)
        ax.set_xlabel(r"Updates", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend()
        # plt.show()
        plt.savefig(directory_path_name + '/Energy-network-' + name + '.png', dpi=300)
        plt.close(fig)

    # --------------- PLOT: pattern of w-matrices --------------- #
    w_mat_plot = np.asarray(w_mat).reshape((30, 30))  # reshape pattern
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(w_mat_plot, interpolation='nearest', cmap='viridis')
    cax.set_clim(vmin=0, vmax=1)
    cbar = fig.colorbar(cax, ticks=[0, 0.3, 0.5, 1], orientation='vertical')
    # plt.show()
    plt.savefig(directory_path_name + '/w-matrices.png', dpi=300)
    plt.close(fig)

    # --------------- PLOT: energy of network during updates --------------- #
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.set_title(f'Energy of network after updates', fontsize=22, loc='left')
    # data
    for j in range(0, len(E_all)):
        # control name
        name = test_mat_names[j]
        # energy of whole network after updating the network.
        Energy_network_list = E_all[j]
        steps_list = np.arange(0, len(Energy_network_list), 1)
        ax.plot(steps_list, Energy_network_list, '-o', label=f"Energy of network: {name}")
    # description
    ax.set_ylabel(r"$E_{network}$", fontsize=18)
    ax.set_xlabel(r"Updates", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    # plt.show()
    plt.savefig(directory_path_name + '/Energy-network-all.png', dpi=300)
    plt.close(fig)
