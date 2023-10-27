import numpy as np
# from pprint import pprint
import matplotlib.pyplot as plt
from Task_1 import hopfield_network, test_network, make_directory

# --------------- TRAINING MATRICES --------------- #
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

# training_mat = [mat_zero, mat_one, mat_two]

# --------------- INPUT MATRICES: PRESENTS HOW TRAINING LOOKS --------------- #
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

# --------------------------------- TRAINING: ADD NEW VECTORS
training_two_1 = np.matrix([
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, 1, -1,
    -1, 1, 1, -1, -1,
    1, -1, -1, -1, -1,
    1, 1, 1, 1, 1,
])

training_two_2 = np.matrix([
    1, 1, 1, -1, -1,
    -1, -1, -1, 1, -1,
    -1, -1, -1, 1, -1,
    -1, 1, 1, -1, -1,
    1, -1, -1, -1, -1,
    1, -1, -1, -1, -1,
])

# pattern_plot = np.asarray(training_zero_2).reshape((6, 5))  # reshape pattern
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.imshow(pattern_plot, interpolation='nearest', cmap='viridis')
# cax.set_clim(vmin=0, vmax=1)
# cbar = fig.colorbar(cax, ticks=[0, 0.3, 0.5, 1], orientation='vertical')
# plt.show()

test_mats = [mat_noisy0, mat_noisy2, mat_noisy2b,
             training_two_1, training_two_2]
test_mat_names = ["noisy0", "noisy2", "noisy2b",
                  "noisy2_added_firsts", "noisy2_added_second"]

# --------------- UPDATE TRAINING MATRICES --------------- #
"""
I notice that my Hopfield network dealing good with that matrices:

    mat_noisy0, mat_noisy2: good recognition.

    mat_noisy2b: bad recognition.
    
So my first ideal is just taken that matrices, which my network
have do right recognition. In another words I just add to my
network those matrices, which my network can dealing.

One important note: if I add that matrices I have to remember that
I have to feed my network with higher number of truly right matrices.
So my whole network will contain for example:

    3 x mat_zero, 3 x mat_one, 3 x mat_two, mat_noisy0, mat_noisy2
"""
n_times_true_one = 6  # how many times I add correct matrices
n_time_zeros = 1  # how many times I add another matrices
n_time_another = 2  # how many times I add another matrices
new_training_mat = []

# appending: true correct matrices
for times in range(0, n_times_true_one):
    new_training_mat.append(mat_zero)
    new_training_mat.append(mat_one)
    new_training_mat.append(mat_two)

for times in range(0, n_time_another):
    # two
    new_training_mat.append(training_two_1)
    new_training_mat.append(training_two_2)

# --------------- CHECK THE RESULTS --------------- #
if __name__ == '__main__':
    directory_path_name = "./Task3"
    make_directory(directory_path_name)
    w_mat = hopfield_network(new_training_mat)  # results of our training
    n_test = len(test_mats)
    Energy_network_list = []
    E_all = []

    print("len(test_mat_names):", len(test_mat_names))
    print("len(test_mats):", len(test_mats))

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
        test_mat, Energy_network_list = test_network(test_mats[i], w_mat, steps=10)
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
