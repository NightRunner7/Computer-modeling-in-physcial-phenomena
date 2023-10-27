import numpy as np
import matplotlib.pyplot as plt
import os
# localization
currentdir = os.getcwd()

# ------------------------ MAKE GIF ------------------------ #
import glob
from PIL import Image

def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.PNG")]
    frame_one = frames[0]
    frame_one.save(f"{frame_folder}/Evolution-Grey-Scott-1D.gif", format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)

# ------------------------ DIFFUSION IN 1D ------------------------ #
class Gray_Scott_1D(object):
    def __init__(self, u_list, v_list, parameters):
        """
        :param u_list: list of one species, which inhibit himself. One important remark:
            this list should be 1D matrix.
            (array or list)
        :param v_list: list of one species, which catalyses himself. One important remark:
            this list should be 1D matrix.
            (array or list)
        :param parameters: list of values of parameters, which are necessary in our case.

        This version should be faster during `Euler_evolve()` calculation uses np.roll()
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
    def compute_laplacian(self, f_list):
        """
        realisation of:

            f''(x_{i}) = (f(x_{i+1}) + f(x_{i-1}) - 2 * f(x_{i}) ) / dx^2

        So,

            f''(x_{i}, y_{i}) = f''(x_{i}) + f''(y_{i})

        :param f_list: the fi values ( fi == f(x_{i} ).
            (array or list)
        :return: laplacian to the input list: f''i.
            (array)

        1D situation!
        """
        L = f_list
        l_i_j = np.roll(np.roll(L, 0, axis=0), 0, axis=1)
        # same j-index
        L_im1_j = np.roll(np.roll(L, -1, axis=0), 0, axis=1)
        l_ip1_j = np.roll(np.roll(L, +1, axis=0), 0, axis=1)
        # compute Laplacian
        lx = (l_ip1_j + L_im1_j - 2 * l_i_j) / self.dx ** 2
        Delta_f = lx

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
        # before step
        u = np.array(self.u, dtype=np.float64)
        v = np.array(self.v, dtype=np.float64)
        # compute laplacian
        Delta_u = self.compute_laplacian(u)
        Delta_v = self.compute_laplacian(v)

        for i in range(0, self.Nrow):
            for j in range(0, self.Ncol):
                # do a one step (in time)
                u[i][j] += (Du * Delta_u[i][j] - u[i][j] * (v[i][j]) ** 2 + F * (1.0 - u[i][j])) * dt
                v[i][j] += (Dv * Delta_v[i][j] + u[i][j] * (v[i][j]) ** 2 - (F + k) * v[i][j]) * dt

        # after step
        self.u = np.array(u, dtype=np.float64)
        self.v = np.array(v, dtype=np.float64)

    # ------------------------------- OTHERS -------------------------------#
    def Back_to_initial(self):
        """
        Back to the initial condition
        """
        self.u = self.u0
        self.v = self.v0

    def Return_u(self):
        return self.u

    def Return_v(self):
        return self.v

# ------------------- INITIAL CONDITION -------------------#
# initial condition
N = 100
dx = 0.02
x_list = np.linspace(0, 2.0, N)

dt = 1.0
Du = 2 * 1e-5
Dv = 1 * 1e-5
F = 0.025
k = 0.055
parameters = [dx, dt, Du, Dv, F, k]
# path
output_plots_path = currentdir + '/1Dvol2_Du_' + str(Du) + '_Dv_' + str(Dv) + '_F_' + str(F) + '_k_' + str(k)

# initial ux and vx
u = np.ones(N)
v = np.zeros(N)
xs = np.arange(N)
for i in range(int(N / 4), int(3 * N / 4)):
    u[i] = np.random.random() * 0.2 + 0.4
    v[i] = np.random.random() * 0.2 + 0.2


GrayScott = Gray_Scott_1D(u, v, parameters)
# ------------------- PLOT EVOLUTION -------------------#
NtSteps = 5000
time_list = np.array([])
all_maximum_points = np.array([])
flag_maximum = 0.2 # maybe in different cases you have to change it.

try:
    # Create target Directory
    os.mkdir(output_plots_path)
    print("Directory ", output_plots_path,  " Created ")
except FileExistsError:
    print("Directory ", output_plots_path,  " already exists")


for step in range(1, NtSteps):
    # do a one step of evolution
    print(f'Im doing step {step}')
    GrayScott.Euler_evolve()
    # return v-list
    new_v = GrayScott.Return_v()

    # position of maxima
    quntity_of_maxima = 0
    for i in range(0,N):
        if new_v[i] > flag_maximum:
            all_maximum_points = np.append(all_maximum_points, x_list[i])
            quntity_of_maxima += 1
    # update the time
    time = [step * dt] * quntity_of_maxima
    time_list = np.append(time_list, time)
    # print('localMax_v_index:', localMax_v_index)

    if step % 10 == 0:
        # return u-list
        new_u = GrayScott.Return_u()

        # --- plotting
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

# make gif
make_gif(output_plots_path)

# ------------------- EVOLUTION OF MAXIMA -------------------#
fig, ax = plt.subplots(figsize=(9.0, 6.0))
ax.set_title(f'Evolution of v(x) maxima', fontsize=22, loc='left')
# data
ax.plot(all_maximum_points, time_list, '.', label=r"Field: $v$")
# description
ax.set_ylabel(r"$time$ [dimensionless]", fontsize=18)
ax.set_xlabel(r"$Space$ [dimensionless]", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
# plt.show()
plt.savefig(output_plots_path + '/Evolution-position-v-maxima.png')
plt.close(fig)