import numpy as np
import os
# from pprint import pprint
import matplotlib.pyplot as plt

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
    for j in range(0, n_matrix):
        # create training pattern
        matrix = training_matrices[j]
        pattern = matrix.flatten()
        # update network
        w_matrix = w_matrix + np.outer(pattern, pattern) / n_matrix

    # Subtract identity matrix
    identity_matrix = np.identity(size_matrix)
    w_matrix = w_matrix - identity_matrix

    return np.matrix(w_matrix)

def test_network(pattern, w_matrix, steps=10):
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
        for j in range(0, n_elements):
            x_i = pattern[0, j]
            # calculate: sum over j of `w_ij * x_j`
            h_i = w_matrix[j].dot(pattern.T)
            # compute energy
            E_i = float(- 0.5 * x_i * h_i)
            E_list.append(E_i)
        # update test matrix
        pattern = np.sign(w_matrix.dot(pattern.T).T)
        E_network = sum(E_list)
        E_network_list.append(E_network)

    return pattern, E_network_list

# --------------- CHECK THE RESULTS --------------- #
if __name__ == '__main__':
    directory_path_name = "./Task1"
    make_directory(directory_path_name)
    w_mat = hopfield_network(training_mat)  # results of our training
    n_test = len(test_mats)
    Energy_network_list = []
    E_all = []

    for i in range(0, n_test):
        # control name
        name = test_mat_names[i]

        # --- PLOT: starting-configuration
        pattern_plot = np.asarray(test_mats[i]).reshape((6, 5))  # reshape pattern

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(pattern_plot, interpolation='nearest', cmap='viridis')
        cax.set_clim(vmin=0, vmax=1)
        cbar = fig.colorbar(cax, ticks=[0, 0.3, 0.5, 1], orientation='vertical')
        # plt.show()
        plt.savefig(directory_path_name + '/starting-configuration-' + name + '.png', dpi=300)
        plt.close(fig)

        # --- do evolution of pattern configuration
        test_mat, Energy_network_list = test_network(test_mats[i], w_mat)
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
        ax.plot(steps_list, Energy_network_list, '-o', color='tab:brown', label=f"initial configuration: {name}")
        # description
        ax.set_ylabel(r"$E_{network}$", fontsize=18)
        ax.set_xlabel(r"time", fontsize=18)
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
    ax.set_title(f'Energy evolution', fontsize=22, loc='left')
    # data
    for i in range(0, len(E_all)):
        # control name
        name = test_mat_names[i]
        # energy of whole network after updating the network.
        Energy_network_list = E_all[i]
        steps_list = np.arange(0, len(Energy_network_list), 1)
        ax.plot(steps_list, Energy_network_list, '-o', label=f"initial configuration: {name}")
    # description
    ax.set_ylabel(r"$E_{network}$", fontsize=18)
    ax.set_xlabel(r"time", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    # plt.show()
    plt.savefig(directory_path_name + '/Energy-network-all.png', dpi=300)
    plt.close(fig)
