import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from numpy.linalg import norm

arr = np.array([3, 4])
norm_l1 = norm(arr)
print("norm_l1:", norm_l1)

# input
Nx = 10  # 520
Ny = 5  # 180
u_in = 0.04
Re = 1000
# create
x_arr = np.arange(0, Nx, 1)
y_arr = np.arange(0, Ny, 1)
nu_LB = u_in * Ny / (2 * Re)  # viscosity
tau = 3 * nu_LB + 0.5  # relaxation time

# vector u0
epsilon = 0.0001  # velocity perturbation
vector_zero = [0.0, 0.0]
u0 = np.array([x[:] for x in [[vector_zero] * Nx] * Ny])

print("u0[:, 1]")
pprint(u0[:, 1])
print("u0")
pprint(u0)

for i in range(0, Ny):
    val_u0_i = u_in * (1 + epsilon * np.sin(y_arr[i] / (2 * np.pi * (Ny - 1))))
    u0_i = [1 * val_u0_i, 0 * val_u0_i]
    # update
    u0[i, :] = u0_i

print("u0")
pprint(u0)

# ------------------ check of cross multiplication
numbers = np.ones((Nx, Ny))
print("size: numbers", numbers.shape)

e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])

vectors = np.array([x[:] for x in [[vector_zero] * Ny] * Nx])  # velocity vector with components (x,y)
vectors[:, :] = numbers[:, :, np.newaxis] * e[1]

print("----------------------------------------------------------------")
print("vectors")
pprint(vectors)
print("vectors shape:", vectors.shape)
print("----------------------------------------------------------------")




# -------------------
u0_x = u0[:, :, 0]
u0_y = u0[:, :, 1]

pprint(u0_x)


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(u0_x, interpolation='nearest', cmap='hot')
# cax.set_clim(vmin=0, vmax=2)
# cbar = fig.colorbar(cax, ticks=[0.0, 0.3, 0.7, 1.0], orientation='vertical')
plt.show()

# Nx = 10
# Ny = 5
# f_eq = np.array(9 * [[x[:] for x in [[1.0] * Nx] * Ny]])
#
# pprint(f_eq)


wedge = np.fromfunction(lambda x, y: (abs(x - Nx / 4) + abs(y)) < Ny / 2, (Nx, Ny), dtype=int)
pprint(wedge)


numbers_vol2 = np.where(wedge == True, 10, numbers)

print("--------------------------------------------------------")
print("numbers")
pprint(numbers)
print()
print("numbers_vol2")
pprint(numbers_vol2)
print("--------------------------------------------------------")


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(wedge.T, interpolation='nearest', cmap='hot')
# cax.set_clim(vmin=0, vmax=2)
# cbar = fig.colorbar(cax, ticks=[0.0, 0.3, 0.7, 1.0], orientation='vertical')
plt.show()

e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])

for i in range(0, 8):
    print("i:", i)