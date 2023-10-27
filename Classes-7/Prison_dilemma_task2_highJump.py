import numpy as np
from pprint import pprint
from Prison_dilema import PrisonDilemma
import matplotlib.pyplot as plt

# --------------- HELPFUL FUNCTION --------------- #
def get_index_2d(flat_index, _n_size):
    """
    :param flat_index: index of flat matrices.
        (array)
    :param _n_size: number of rows and columns. We're going to investigate only
        square matrices.
        (float)
    :return: index, which is corresponding to 2D matrix
    """
    col_index = flat_index % _n_size
    row_index = int(flat_index / _n_size)
    return row_index, col_index

def create_initial_lattice(_n_size):
    """
    This function will create a lattice which create a grid, where 50% of players
    are cooperators and 50% are defectors.

    :param _n_size: the size (dimension) of square lattice of our interest
        (array: square)
    :return: lattice with cooperators and defectors.
        (array: square)
    """
    init_lattice = np.array([[None for x in range(n_size)] for y in range(n_size)])
    flat_indexes_defectors = np.random.choice(n_size * n_size, int(n_size * n_size / 2), replace=False)
    for defector in range(0, len(flat_indexes_defectors)):
        index = flat_indexes_defectors[defector]
        index_i, index_j = get_index_2d(index, n_size)
        # set defector
        init_lattice[index_i][index_j] = 'D'

    # set cooperators
    init_lattice = np.where(init_lattice == None, 'C', init_lattice)

    return init_lattice

# ----------------------- PREPARE DATA ----------------------- #
# --- initial sets
steps = 100
n_size = 201
# --- set the b_val
val_b_add_part_1 = np.linspace(1.12, 1.13, num=100, endpoint=True)
val_b_add_part_2 = np.linspace(1.28, 1.29, num=100, endpoint=True)
val_b_add_part_3 = np.linspace(1.49, 1.50, num=100, endpoint=True)
val_b_add_part_4 = np.linspace(1.59, 1.62, num=400, endpoint=True)
val_b_add_part_5 = np.linspace(1.79, 1.81, num=400, endpoint=True)
val_b_add_part_6 = np.linspace(1.99, 2.0, num=400, endpoint=True)

list_of_lists_val_b = [val_b_add_part_1, val_b_add_part_2, val_b_add_part_3, val_b_add_part_4,
                       val_b_add_part_5, val_b_add_part_6]

# --- data to expose
ratio_defs_equ = np.array([])  # defs: deflectors, equ: equilibrium

# --- create a lattice, where we will have 50% of cooperators and 50% of defectors
init_lattice = create_initial_lattice(n_size)
# --- create a object class, which will dealing with grid
class_PrisonDilemma = PrisonDilemma(init_lattice, val_b=val_b_add_part_1[0])


for j in range(0, len(list_of_lists_val_b)):
    val_b_list = list_of_lists_val_b[j]
    # --- clear data
    ratio_defs_equ = np.array([])

    print('I m working with list numer:', j)

    for val_b in val_b_list:
        # --- set the new grid
        init_lattice = create_initial_lattice(n_size)
        # --- update data stored in class, which dealing with prison dilemma
        class_PrisonDilemma.put_lattice(init_lattice)
        class_PrisonDilemma.update_b_parameter(val_b)

        for i in range(0, steps):
            # --------------------------- STATE OF LATTICE --------------------------- #
            if i == steps - 1:
                ratio_deflectors = class_PrisonDilemma.return_ratio_deflectors()
                ratio_defs_equ = np.append(ratio_defs_equ, ratio_deflectors)

            # --------------------------- DO ONE STEP --------------------------- #
            class_PrisonDilemma.one_step()

    # --------------------------- PLOT: STATE OF LATTICE --------------------------- #
    region = f'[{"{:.3f}".format(val_b_list[0])}, {"{:.2f}".format(val_b_list[-1])}]'
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.set_title(r'Ratio of deflectors, equilibrium in: ' + region, fontsize=22, loc='left')
    # data
    ax.plot(val_b_list, ratio_defs_equ, label=f"equilibrium state, steps: {100}")
    # description
    ax.set_ylabel(r"$R_{deflectors}$", fontsize=18)
    ax.set_xlabel(r"'DC' payoff", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    # plt.show()
    plt.savefig(f'./-Equilibrium-state-of-deflectors-' + region + '-.png', dpi=300)
    plt.close(fig)
