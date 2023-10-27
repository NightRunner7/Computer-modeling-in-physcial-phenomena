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
    frame_one.save(f"{frame_folder}/Evolution-Grey-Scott-2D.gif", format="GIF", append_images=frames,
                   save_all=True, duration=30, loop=0)

# ------------------------------- CLASS DEFINITION -------------------------------#
class Gray_Scott_2D(object):
    def __init__(self, u_list, v_list, parameters):
        """
        :param u_list: list of one species, which inhibit himself. One important remark:
            this list should be 2D matrix: NxN - square matrix.
            (array or list)
        :param v_list: list of one species, which catalyses himself. One important remark:
            this list should be 2D matrix: NxN - square matrix.
            (array or list)
        :param parameters: list of values of parameters, which are necessary in our case.

        This version should be faster during `Euler_evolve()` calculation uses np.roll()
        """
        # parameters
        self.dx = parameters[0]
        self.dy = parameters[1]
        self.dt = parameters[2]
        self.Du = parameters[3]
        self.Dv = parameters[4]
        self.F = parameters[5]
        self.k = parameters[6]
        # initial
        self.u0 = np.array(u_list, dtype=np.float64)
        self.v0 = np.array(v_list, dtype=np.float64)
        self.Nrow = len(self.u0)  # Number of rows: i
        self.Ncol = len(self.u0[0])  # Number of columns: j
        # present
        self.u = np.array(u_list, dtype=np.float64)
        self.v = np.array(v_list, dtype=np.float64)

    # ------------------------------- NECESSARY fUNCTIONS -------------------------------#
    def compute_laplacian(self, f_list):
        """
        realisation of:

            f''(x_{i}) = (f(x_{i+1}) + f(x_{i-1}) - 2 * f(x_{i}) ) / dx^2
            f''(y_{i}) = (f(y_{i+1}) + f(y_{i-1}) - 2 * f(y_{i}) ) / dy^2

        So,

            f''(x_{i}, y_{i}) = f''(x_{i}) + f''(y_{i})

        :param f_list: the fi values ( fi == f(x_{i}) ).
            (array or list)
        :return: laplacian to the input list: f''i.
            (array)

        2D situation!
        """
        L = f_list
        l_i_j = np.roll(np.roll(L, 0, axis=0), 0, axis=1)
        # same j-index
        L_im1_j = np.roll(np.roll(L, -1, axis=0), 0, axis=1)
        l_ip1_j = np.roll(np.roll(L, +1, axis=0), 0, axis=1)
        # same i-index
        l_i_jm1 = np.roll(np.roll(L, 0, axis=0), -1, axis=1)
        l_i_jp1 = np.roll(np.roll(L, 0, axis=0), +1, axis=1)
        # compute Laplacian
        lx = (l_ip1_j + L_im1_j - 2 * l_i_j) / self.dx ** 2
        ly = (l_i_jp1 + l_i_jm1 - 2 * l_i_j) / self.dy ** 2
        Delta_f = lx + ly

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
dy = 0.02
x_list = np.arange(N)
y_list = np.arange(N)

dt = 1.0
Du = 2*1e-5
Dv = 1*1e-5
# values of F
F_start = 0.022
F_end = 0.09
F_nums = 18
F_list = np.linspace(F_start, F_end, F_nums)
for i in range(0, F_nums):
    napis = f'{F_list[i]:8.4f}'
    print(napis, end=" ")
print("F_list:", F_list)
# fixed value of k
k = 0.0630

parameters = [dx, dy, dt, Du, Dv, F_start, k]

u = np.ones((N, N))
v = np.zeros((N, N))
for i in range(int(N/4), int(3*N/4)):
    for j in range(int(N/4), int(3*N/4)):
        u[i][j] = np.random.random()*0.2+0.4
        v[i][j] = np.random.random()*0.2+0.2

# print('u:',u)
# print('v:', v)
output_plots_path = currentdir + '/Task3_vol2_2D_Du_' + str(Du) + '_Dv_' + str(Dv) + '_F_' + '_k_' + str(k)

try:
    # Create target Directory
    os.mkdir(output_plots_path)
    print("Directory ", output_plots_path,  " Created ")
except FileExistsError:
    print("Directory ", output_plots_path,  " already exists")

# ------------------- PLOT EVOLUTION -------------------#
GrayScott = Gray_Scott_2D(u, v, parameters)
NtSteps = 7000
# NtSteps = 1

for i in range(0, F_nums):
    Fvalue = F_list[i]
    F_napis = f'{F_list[i]:.4f}'
    # set the proper value of F
    GrayScott.F = Fvalue
    # back to initial condition
    # GrayScott.Back_to_initial()

    print(f'Im doing Fvalue ' + F_napis)

    for step in range(1, NtSteps):
        # do a one step of evolution
        # print(f'Im doing step {step}')
        GrayScott.Euler_evolve()

        if step % 100 == 0:
        # if step > NtSteps-2:
            # I have to change matrix from 1D into 2D
            u_2D = GrayScott.Return_u()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.imshow(u_2D, interpolation='nearest', cmap='viridis')
            cax.set_clim(vmin=0, vmax=1)
            cbar = fig.colorbar(cax, ticks=[0, 0.3, 0.5, 1], orientation='vertical')

            napis = f'{step:05d}'
            plt.savefig(output_plots_path + '/' + 'Gray-Scott-2D-F-' + F_napis + '-in-step-' + napis + '.png', dpi=300)
            plt.close(fig)

# ------------------- CREATE A GIF -------------------#
# make gif
make_gif(output_plots_path)
