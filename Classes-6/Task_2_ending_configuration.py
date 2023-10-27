from Task_2 import *
from pprint import pprint

# test_mats = [mat_noisy0, mat_noisy2, mat_noisy2b]
# test_mat_names = ["noisy0", "noisy2", "noisy2b"]
test_mats = [mat_noisy2b]
test_mat_names = ["noisy2b"]

directory_path_name = "./Task2"
make_directory(directory_path_name)
w_mat = hopfield_network(training_mat)  # results of our training
n_test = len(test_mats)
n_configuration = 100
Energy_network_list = []

for j in range(0, n_test):
    # control name
    name = test_mat_names[j]
    new_dir_path = directory_path_name + '/' + name
    make_directory(new_dir_path)

    # deal with ending states
    unique_end_states = 0
    unique_E_end = []
    unique_E_evolution = []
    unique_E_number = []

    # all Energy_network_list
    all_E_network = []

    for configuration in range(0, n_configuration):
        # --- do evolution of pattern configuration
        test_mat, Energy_network_list = test_network_asynchronously(test_mats[j].copy(), w_mat, steps=10)
        steps_list = np.arange(0, len(Energy_network_list), 1)
        # append
        all_E_network.append(Energy_network_list)

        # --- unique ending energy states
        normal_ending_state = any(item in unique_E_end for item in unique_E_end if item == Energy_network_list[-1])
        if normal_ending_state is False:
            unique_end_states += 1
            unique_E_end.append(Energy_network_list[-1])
            unique_E_evolution.append(Energy_network_list)
            unique_E_number.append(configuration)

        # --- PLOT: ending configuration pattern
        pattern_plot = np.asarray(test_mat).reshape((6, 5))  # reshape pattern

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(name + r', configuration: ' + f'{configuration:03d}', fontsize=15, loc='left')
        cax = ax.imshow(pattern_plot, interpolation='nearest', cmap='viridis')
        cax.set_clim(vmin=0, vmax=1)
        cbar = fig.colorbar(cax, ticks=[0, 0.3, 0.5, 1], orientation='vertical')
        # plt.show()
        plt.savefig(new_dir_path + '/' + name + f'-ending-configuration-{configuration:03d}.png', dpi=300)
        plt.close(fig)

        # --- PLOT: energy of network during updates
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        ax.set_title(r'Energy of network for: ' + name, fontsize=22, loc='left')
        # data
        ax.plot(steps_list, Energy_network_list, '-o', color='tab:brown', label=f"configuration: {configuration}")
        # description
        ax.set_ylabel(r"$E_{network}$", fontsize=18)
        ax.set_xlabel(r"time", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend()
        # plt.show()
        plt.savefig(new_dir_path + '/' + name + f'-Energy-network-ending-configuration-{configuration:03d}.png')
        plt.close(fig)

    print("unique_end_states:", unique_end_states)
    # --- PLOT: energy of network during updates
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.set_title(r'Energy of network for: ' + name, fontsize=22, loc='left')
    # data
    for unique_state in range(0, unique_end_states):
        Energy_network_list = unique_E_evolution[unique_state]
        steps_list = np.arange(0, len(Energy_network_list), 1)
        conf = unique_E_number[unique_state]  # about what configuration we're talking about
        ax.plot(steps_list, Energy_network_list, '-o', label=f"configuration: {conf}")
    # description
    ax.set_ylabel(r"$E_{network}$", fontsize=18)
    ax.set_xlabel(r"time", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    # plt.show()
    plt.savefig(new_dir_path + '/' + name + f'-Energy-network-ending-configuration.png', dpi=300)
    plt.close(fig)

# pprint(all_E_network)
