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

# ------------------------------- HELPFUL FUNCTION -------------------------------#
def get_matrix(_u, _nx, _ny):
    """
    this function deflatten a vector
    so that we can get a nx\times ny matrix
    """
    matrix = np.asarray(_u).reshape((_ny,_ny))
    return matrix

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
        self.u0 = np.array(u_list.flatten(), dtype=np.float64)
        self.v0 = np.array(v_list.flatten(), dtype=np.float64)
        self.N = len(u_list)
        self.N2 = len(u_list) ** 2
        # present
        self.u = np.array(u_list.flatten(), dtype=np.float64)
        self.v = np.array(v_list.flatten(), dtype=np.float64)

    # ------------------------------- NECESSARY fUNCTIONS -------------------------------#
    def periodic_boundary_condition_2D(self, i, j):
        """
        :param j: index of list element, which we consider during
            doing calculation: one of index of square matrix.
            (float)
        :param i: index of list element, which we consider during
            doing calculation: one of index of square matrix.
            (float)
        :return: the index after applying periodic boundary condition.
            In the end we land with 1D array or list!
            (float)
        """
        if (i < 0):
            i += self.N
        elif i == self.N:
            i += - self.N

        if (j < 0):
            j += self.N
        elif j == self.N:
            j += - self.N
        return i + j * self.N

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
        """
        Delta_f = np.zeros(self.N2)

        for i in range(0, self.N):
            for j in range(0, self.N):
                ip1 = self.periodic_boundary_condition_2D(i + 1, j)
                im1 = self.periodic_boundary_condition_2D(i - 1, j)
                jp1 = self.periodic_boundary_condition_2D(i, j + 1)
                jm1 = self.periodic_boundary_condition_2D(i, j - 1)
                index = self.periodic_boundary_condition_2D(i, j)

                lx = (f_list[ip1] + f_list[im1] - 2 * f_list[index]) / self.dx ** 2
                ly = (f_list[jp1] + f_list[jm1] - 2 * f_list[index]) / self.dy ** 2

                Delta_f[index] = lx + ly

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

        for index in range(0, self.N2):
            u[index] += (Du * Delta_u[index] - u[index] * (v[index]) ** 2 + F * (1.0 - u[index])) * dt
            v[index] += (Dv * Delta_v[index] + u[index] * (v[index]) ** 2 - (F + k) * v[index]) * dt

        # after step
        self.u = np.array(u, dtype=np.float64)
        self.v = np.array(v, dtype=np.float64)

        return [self.u, self.v]

    # ------------------------------- OTHERS -------------------------------#
    def Back_to_initial(self):
        """
        Back to the initial condition
        """
        self.u = self.u0
        self.v = self.v0

    def Give_u_v_flatten(self):
        """
        :return: list u and v as 1D array.
        """
        return [self.u, self.v]

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
F_list = [0.025, 0.03, 0.01, 0.04, 0.06, 0.037]
k_list = [0.055, 0.062, 0.047, 0.07, 0.0615, 0.06]
F = F_list[2]
k = k_list[2]
parameters = [dx, dy, dt, Du, Dv, F, k]

u = np.ones((N, N))
v = np.zeros((N, N))
for i in range(int(N/4), int(3*N/4)):
    for j in range(int(N/4), int(3*N/4)):
        u[i][j] = np.random.random()*0.2+0.4
        v[i][j] = np.random.random()*0.2+0.2

# print('u:',u)
# print('v:', v)

output_plots_path = currentdir + '/2D_Du_' + str(Du) + '_Dv_' + str(Dv) + '_F_' + str(F) + '_k_' + str(k)
try:
    # Create target Directory
    os.mkdir(output_plots_path)
    print("Directory ", output_plots_path,  " Created ")
except FileExistsError:
    print("Directory ", output_plots_path,  " already exists")

# ------------------- PLOT EVOLUTION -------------------#
GrayScott = Gray_Scott_2D(u, v, parameters)
NtSteps = 5000
# NtSteps = 1

for step in range(1, NtSteps):
    # do a one step of evolution
    print(f'Im doing step {step}')
    [new_u, new_v] = GrayScott.Euler_evolve()

    if step % 100 == 0:
        # I have to change matrix from 1D into 2D
        u_2D = get_matrix(new_u, N, N)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(u_2D, interpolation='nearest', cmap='viridis')
        cax.set_clim(vmin=0, vmax=1)
        cbar = fig.colorbar(cax, ticks=[0, 0.3, 0.5, 1], orientation='vertical')
        if step < 100:
            napis = '00' + str(step)
        elif step < 1000:
            napis = '0' + str(step)
        else:
            napis = str(step)
        plt.savefig(output_plots_path + '/' + 'Gray-Scott-2D-system-in-step-' + napis + '.png', dpi=300)
        plt.close(fig)

# ------------------- CREATE A GIF -------------------#
# make gif
make_gif(output_plots_path)
