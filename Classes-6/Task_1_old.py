import numpy as np
import matplotlib.pyplot as plt

def get_matrix(_u, _nx, _ny):
    """
    this function deflatten a vector
    so that we can get a nx\times ny matrix
    """
    matrix = np.asarray(_u).reshape((_ny, _ny))
    return matrix

# --- set matrices
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

# --- input matrices

noisy0 = np.matrix([
    -1, 1, 1, 1, -1,
    1, -1, -1, -1, -1,
    1, -1, -1, -1, 1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, 1, -1, -1,
])

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


def find_w(stored_matrices):
    """
    :param stored_matrices:
    :return:
    """
    N = len(stored_matrices)
    size = len(stored_matrices[0])
    # print("N", N)

    mat_sum = np.zeros((size, size))
    for i in range(0, N):
        print("i", i)
        matrix = stored_matrices[i]
        vector = matrix.flatten()
        outer_products = np.outer(vector, vector)
        # outer_products = np.tensordot(vector, vector)

        # convert into matrix
        # outer_products = np.matrix(outer_products)
        # print(outer_products)
        # print("type:", outer_products.shape)

        mat_sum = mat_sum + outer_products / N

    new_size = len(mat_sum)
    mat_identity = np.identity(new_size)

    # remove identity
    return mat_sum - mat_identity

# --- find matrices w
matrices_w = np.matrix(find_w([mat_zero, mat_one, mat_two]))

# mat = np.matrix([[1, 0],
#                      [0, 1]])
#
# # numpy_one = np.squeeze(np.asarray(mat_one))
#
# vector = np.matrix([1, 1])
#
# # print(numpy_one)
# print(mat)
# print(vector)
#
# # print(mat.dot(vector.T))
# step = mat.dot(vector.T)
# print(vector.dot(step))

steps = 7
mat_inp = mat_noisy2b
Energy_list = []

for step in range(0, steps):
    vector_product = matrices_w.dot(mat_inp.T)
    energy = float(- 0.5 * mat_inp.dot(vector_product))
    # append energy
    Energy_list.append(energy)
    # do one step
    mat_inp = np.sign(matrices_w.dot(mat_inp.T).T)

print(Energy_list)

matrix_plot = np.asarray(mat_inp).reshape((6, 5))

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(matrices_w, interpolation='nearest', cmap='viridis')
cax.set_clim(vmin=0, vmax=1)
cbar = fig.colorbar(cax, ticks=[0, 0.3, 0.5, 1], orientation='vertical')
plt.show()
# plt.savefig("./noise2b", dpi=300)
# plt.close(fig)
