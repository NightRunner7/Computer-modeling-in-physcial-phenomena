import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import imageio.v2 as imageio
from pathlib import Path

import os
# localization
currentdir = os.getcwd()

class Gray_Scott_1D(object):
    def __init__(self, u_list, v_list, parameters):
        """
        :param u_list:
        :param v_list:
        :param parameters:
        """
        # parameters
        self.dx = parameters[0]
        self.dt = parameters[1]
        self.Du = parameters[2]
        self.Dv = parameters[3]
        self.F = parameters[4]
        self.k = parameters[5]
        # initial
        self.u0 = np.array(u_list, dtype=np.float64)
        self.v0 = np.array(v_list, dtype=np.float64)
        self.N = len(u_list)
        # present
        self.u = np.array(u_list, dtype=np.float64)
        self.v = np.array(v_list, dtype=np.float64)

    # ------------------------------- NECESSARY fUNCTIONS -------------------------------#
    def periodic_boundary_condition_1D(self, index):
        """
        :param index: index of list element, which we consider during
            doing calculation.
            (float)
        :return: the index after applying periodic boundary condition.
            (float)
        """
        if (index < 0):
            index = index + self.N
        elif index == self.N:
            index = index - self.N
        return index

    def compute_laplacian(self, f_list):
        """
        realisation of:

            f''(x_{i}) = (f(x_{i+1}) + f(x_{i-1}) - 2 * f(x_{i}) ) / dx^2

        :param f_list: the fi values ( fi == f(x_{i}) ).
            (array or list)
        :return: laplacian to the input list: f''i.
            (list)
        """
        Delta_f = np.zeros(self.N)

        for index in range(0, self.N):
            ip1 = self.periodic_boundary_condition_1D(index + 1)
            im1 = self.periodic_boundary_condition_1D(index - 1)

            Delta_f[index] = (f_list[ip1] + f_list[im1] - 2 * f_list[index]) / self.dx**2

        return Delta_f

    def Euler_evolve(self):
        """
        :return: u and v list after one Euler evolution step.
        """
        # parameters
        Du = self.Du
        Dv = self.Dv
        F = self.F
        k = self.k
        dt = self.dt
        # laplacian
        Delta_u = self.compute_laplacian(self.u)
        Delta_v = self.compute_laplacian(self.v)
        # before step
        u = np.array(self.u, dtype=np.float64)
        v = np.array(self.v, dtype=np.float64)

        for index in range(0, self.N):
            u[index] += (Du * Delta_u[index] - u[index] * (v[index])**2 + F * (1.0 - u[index])) * dt
            v[index] += (Dv * Delta_v[index] + u[index] * (v[index])**2 - (F + k) * v[index]) * dt

        # after step
        self.u = np.array(u, dtype=np.float64)
        self.v = np.array(v, dtype=np.float64)

        return [self.u, self.v]

# ------------------- INITIAL CONDITION -------------------#

# initial condition
N = 100
dx = 0.02
x_list = np.linspace(0, 2.0, N)

dt = 1.0
Du = 2*1e-5
Dv = 1*1e-5
F = 0.025
k = 0.055
parameters = [dx, dt, Du, Dv, F, k]
# path
output_plots_path = currentdir + '/1D_Du_' + str(Du) + '_Dv_' + str(Dv) +'_F_' + str(F) + '_k_' + str(k)

# initial ux and vx
u = np.ones(N)
v = np.zeros(N)
xs= np.arange(N)
for i in range(int(N/4),int(3*N/4)):
    u[i] = np.random.random()*0.2+0.4
    v[i] = np.random.random()*0.2+0.2

# print('u:', u)
# print('v:',v)

fig, ax = plt.subplots(figsize=(9.0, 6.0))
ax.set_title(f'Initial Gray-Scott system', fontsize=22, loc='left')
# data
ax.plot(x_list, u, label=r"Field: $u$")
ax.plot(x_list, v, label=r"Field: $v$")
# description
ax.set_ylabel(r"$v, \, u$ [dimensionless]", fontsize=18)
ax.set_xlabel(r"$t$ [dimensionless]", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
plt.savefig(output_plots_path + '/other' + '/Gray-Scott-1D-initial.png')
plt.close(fig)
# plt.show()

GrayScott = Gray_Scott_1D(u, v, parameters)

# ------------------- PLOT EVOLUTION -------------------#
NtSteps = 5000
all_localMax_v = []
time_list = []

try:
    # Create target Directory
    os.mkdir(output_plots_path)
    print("Directory " , output_plots_path ,  " Created ")
except FileExistsError:
    print("Directory " , output_plots_path ,  " already exists")


for step in range(1, NtSteps):
    # do a one step of evolution
    print(f'Im doing step {step}')
    [new_u, new_v] = GrayScott.Euler_evolve()

    # find in field: `v` local maximum and stored the localization
    # this return the indices
    localMax_v_index = argrelextrema(new_v, np.greater)[0]
    all_localMax_v.extend(localMax_v_index)
    # update the time
    length = len(localMax_v_index)
    time = [step * dt] * length
    time_list.extend(time)
    # print('localMax_v_index:', localMax_v_index)

    if step % 10 == 0:
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        ax.set_title(f'Gray-Scott system in step: {step}', fontsize=22, loc='left')
        # data
        ax.plot(x_list, new_u, label=r"Field: $u$")
        ax.plot(x_list, new_v, label=r"Field: $v$")
        # description
        ax.set_ylabel(r"$v, \, u$ [dimensionless]", fontsize=18)
        ax.set_xlabel(r"$Space$ [dimensionless]", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend()
        # plt.show()
        if step < 100:
            napis = '00' + str(step)
        elif step < 1000:
            napis = '0' + str(step)
        else:
            napis = str(step)
        plt.savefig(output_plots_path + '/' + 'Gray-Scott-system-in-step-' + napis + '.png')
        plt.close(fig)

# ------------------- CREATE A GIF -------------------#
image_path = Path(output_plots_path)
images = list(image_path.glob('*.png'))
image_list = []
for file_name in images:
    image_list.append(imageio.imread(file_name))

# print(image_list)

#creating the gif
gif_name = 'Gray-Scott-1D-F-' + str(F) + '-k-' + str(k) + '.gif'
gif_path = output_plots_path + '/' + gif_name
imageio.mimwrite(gif_path, image_list, fps=5, loop=True)


# ------------------- EVOLUTION OF MAXIMA -------------------#
fig, ax = plt.subplots(figsize=(9.0, 6.0))
ax.set_title(f'Evolution of v(x) maxima', fontsize=22, loc='left')
# data
ax.plot(all_localMax_v, time_list, '.', label=r"Field: $v$")
# description
ax.set_ylabel(r"$time$ [dimensionless]", fontsize=18)
ax.set_xlabel(r"$Space$ [dimensionless]", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
# plt.show()
plt.savefig(output_plots_path + '/other' + '/Evolution-position-v-maxima.png')
plt.close(fig)