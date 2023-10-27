import numpy as np
import matplotlib.pyplot as plt

# Define the system
def f(x, y, z, K):
    x_new = z * np.cos(K * x) + y * np.sin(K * x)
    y_new = y * np.cos(K * x) - z * np.sin(K * x)
    z_new = -x
    return x_new, y_new, z_new

# Set the number of initial conditions and kicks
n_init = 100 # initall condition
n_kicks = 200 # number of kicks

# Generate a set of n_init random initial conditions
x_init = np.random.uniform(-1, 1, size=n_init)
y_init = np.random.uniform(-1, 1, size=n_init)
z_init = np.random.uniform(-1, 1, size=n_init)

# Initialize arrays to store the values of theta and phi at each iteration for each initial condition
theta = np.zeros((n_kicks, n_init))
phi = np.zeros((n_kicks, n_init))

# Plot the values of theta and phi in a phase space plot
fig, ax = plt.subplots(2,2, figsize=(12.0, 12.0))
ax[0,0].set_title('One hundred random initall conditions', fontsize=22, loc='left')

#----- FIRST -----#
for i in range(n_init):
    # Apply the system for n_kicks iterations for each initial condition and store the values of theta and phi
    x = x_init[i]
    y = y_init[i]
    z = z_init[i]
    for j in range(n_kicks):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        phi[j, i] = np.arctan2(y, x)
        theta[j, i] = np.arccos(z / r)
        x, y, z = f(x, y, z, K=1)

    ax[0,0].plot(phi[:,i], theta[:,i], ",")
#--- describing
ax[0,0].set_xlabel(r'$\phi$')
ax[0,0].set_ylabel(r'$\theta$')
ax[0,0].set_xlim(-np.pi, np.pi)
ax[0,0].set_ylim(0, np.pi)

# ----- SECOND -----#
for i in range(n_init):
    # Apply the system for n_kicks iterations for each initial condition and store the values of theta and phi
    x = x_init[i]
    y = y_init[i]
    z = z_init[i]
    for j in range(n_kicks):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        phi[j, i] = np.arctan2(y, x)
        theta[j, i] = np.arccos(z / r)
        x, y, z = f(x, y, z, K=2)

    ax[0, 1].plot(phi[:, i], theta[:, i], ",")
# --- describing
ax[0, 1].set_xlabel(r'$\phi$')
ax[0, 1].set_ylabel(r'$\theta$')
ax[0, 1].set_xlim(-np.pi, np.pi)
ax[0, 1].set_ylim(0, np.pi)

# ----- THIRD -----#
for i in range(n_init):
    # Apply the system for n_kicks iterations for each initial condition and store the values of theta and phi
    x = x_init[i]
    y = y_init[i]
    z = z_init[i]
    for j in range(n_kicks):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        phi[j, i] = np.arctan2(y, x)
        theta[j, i] = np.arccos(z / r)
        x, y, z = f(x, y, z, K=3)

    ax[1, 0].plot(phi[:, i], theta[:, i], ",")
# --- describing
ax[1, 0].set_xlabel(r'$\phi$')
ax[1, 0].set_ylabel(r'$\theta$')
ax[1, 0].set_xlim(-np.pi, np.pi)
ax[1, 0].set_ylim(0, np.pi)

# ----- FOURTH -----#
for i in range(n_init):
    # Apply the system for n_kicks iterations for each initial condition and store the values of theta and phi
    x = x_init[i]
    y = y_init[i]
    z = z_init[i]
    for j in range(n_kicks):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        phi[j, i] = np.arctan2(y, x)
        theta[j, i] = np.arccos(z / r)
        x, y, z = f(x, y, z, K=6)

    ax[1, 1].plot(phi[:, i], theta[:, i], ",")
# --- describing
ax[1, 1].set_xlabel(r'$\phi$')
ax[1, 1].set_ylabel(r'$\theta$')
ax[1, 1].set_xlim(-np.pi, np.pi)
ax[1, 1].set_ylim(0, np.pi)

plt.savefig('Task-3-plot.png')
plt.show()