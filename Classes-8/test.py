import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt


e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
pprint(e[0])

x_list = np.arange(0, 180, 1)
pprint(x_list)

y_arr = np.arange(0, 180, 1)
# print(y_arr)
print("napis:", y_arr[10])


vec_list = []
for i in range(0, 180):
    val_u0_i = (1 + 0.001 * np.sin(y_arr[i] / (2 * np.pi * (180 - 1))))
    u0_i = [1*val_u0_i, 0*val_u0_i]

    vec_list.append(u0_i)

pprint(vec_list)

Nx = 520
x_list = np.arange(0, Nx, 1)
Ny = 180
y_list = np.arange(0, Ny, 1)
lattice = np.zeros((Ny, Nx))

# pprint(lattice)

for iy in range(Ny):
    for ix in range(Nx):
        if (abs(x_list[ix] - Nx/4) + abs(y_list[iy])) < Ny / 2:
            lattice[iy][ix] = 1

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(lattice, interpolation='nearest', cmap='viridis')
cax.set_clim(vmin=0, vmax=1)
cbar = fig.colorbar(cax, ticks=[0.0, 0.3, 0.7, 1.0], orientation='vertical')
plt.show()

# ---------------------------------------------------------------------
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
    This function will create a lattice which create a grid, where 25% of players
    are cooperators, 25% are defectors, 25% are pavlov and 25% are tit-for-tap.

    :param _n_size: the size (dimension) of square lattice of our interest
        (array: square)
    :return: lattice with cooperators and defectors.
        (array: square)
    """
    init_lattice = np.array([[None for x in range(_n_size)] for y in range(_n_size)])
    flat_indexes_defectors = np.random.choice(_n_size * _n_size, _n_size * _n_size, replace=False)

    for player in range(0, int(_n_size ** 2)):
        index = flat_indexes_defectors[player]
        index_i, index_j = get_index_2d(index, _n_size)

        if player < int(_n_size ** 2 / 4):
            # set cooperators
            init_lattice[index_i][index_j] = 'Cop'
        elif player < int(2 * _n_size ** 2 / 4):
            # set deflectors
            init_lattice[index_i][index_j] = 'Def'
        elif player < int(3 * _n_size ** 2 / 4):
            # set pavlov
            init_lattice[index_i][index_j] = 'Pav'
        else:
            # set 'tit-for-tap'
            init_lattice[index_i][index_j] = 'Tit'

    return init_lattice

init_lat = create_initial_lattice(5)
# upper row
l_im1_jm1 = np.roll(np.roll(init_lat, -1, axis=0), -1, axis=1)
l_im1_j = np.roll(np.roll(init_lat, -1, axis=0), 0, axis=1)
l_im1_jp1 = np.roll(np.roll(init_lat, -1, axis=0), +1, axis=1)
# same row
l_i_jm1 = np.roll(np.roll(init_lat, 0, axis=0), -1, axis=1)
l_i_j = np.roll(np.roll(init_lat, 0, axis=0), 0, axis=1)  # HOW WE ARE!
l_i_jp1 = np.roll(np.roll(init_lat, 0, axis=0), +1, axis=1)
# lower row
l_ip1_jm1 = np.roll(np.roll(init_lat, +1, axis=0), -1, axis=1)
l_ip1_j = np.roll(np.roll(init_lat, +1, axis=0), 0, axis=1)
l_ip1_jp1 = np.roll(np.roll(init_lat, +1, axis=0), +1, axis=1)

print("---------------------------------------------------")
print("l_i_j")
pprint(l_i_j)
print()
print("l_ip1_jp1")
pprint(l_i_jm1)

