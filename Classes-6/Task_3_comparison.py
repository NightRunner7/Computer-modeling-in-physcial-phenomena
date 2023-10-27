import numpy as np
# import sys
import matplotlib.pyplot as plt
from Task_1 import hopfield_network, test_network, make_directory, training_mat
from Task_3_newTraining import new_training_mat

"""
Important note: we have two different sets of training matrices:

    training_mat: contains only three matrices (given in task)
    new_training_mat: contains three correct matrices and some
        different one. To more detail go to `Task_3_newTraining`.
"""
# --------------- INPUT MATRICES: PRESENTS HOW TRAINING LOOKS --------------- #
empty = np.matrix([
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1
])
# --- INPUT TWO-LIKE
noise2a = np.matrix([
    1, 1, 1, -1, -1,
    -1, -1, -1, 1, -1,
    -1, -1, -1, 1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1
])

noise2b = np.matrix([
    1, 1, 1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, 1, 1, -1, -1,
    1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1
])

noise2c = np.matrix([
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, 1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, 1, 1, 1
])

noise2d = np.matrix([
    -1, 1, 1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    1, -1, -1, -1, -1,
    -1, -1, -1, 1, -1
])


test_mats = [noise2a, noise2b, noise2c, noise2d]
test_mat_names = ["noise2a", "noise2b", "noise2c", "noise2d"]

# sys.exit()
# --------------- CHECK THE RESULTS --------------- #
model_simple = False  # choose the model
if model_simple is True:
    directory_path_name = "./Task3-comparison-simple"
    make_directory(directory_path_name)
    w_mat = hopfield_network(training_mat)  # results of our training
else:
    directory_path_name = "./Task3-comparison-myModel"
    make_directory(directory_path_name)
    w_mat = hopfield_network(new_training_mat)  # results of our training

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
    ax.set_title("Initial configuration: " + name, fontsize=18, loc='center')
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
    ax.set_title(name, fontsize=18, loc='center')
    cax = ax.imshow(pattern_plot, interpolation='nearest', cmap='viridis')
    cax.set_clim(vmin=0, vmax=1)
    cbar = fig.colorbar(cax, ticks=[0, 0.3, 0.5, 1], orientation='vertical')
    # plt.show()
    plt.savefig(directory_path_name + '/' + name + '.png', dpi=300)
    plt.close(fig)

    # --- PLOT: energy of network during updates
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.set_title(r'Energy of network', fontsize=22, loc='left')
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
