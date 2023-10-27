import numpy as np
from numpy.linalg import norm
from pprint import pprint
import os
import matplotlib.pyplot as plt
import glob
from PIL import Image


# --------------- HELPFUL FUNCTION --------------- #
def make_directory(_path_name):
    """ Create directory with fixed name"""
    try:
        # Create target Directory
        os.mkdir(_path_name)
        print("Directory ", _path_name, " Created ")
    except FileExistsError:
        print("Directory ", _path_name, " already exists")


def make_gif(frame_folder, name):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.PNG")]
    frame_one = frames[0]
    frame_one.save(f"{frame_folder}/{name}.gif", format="GIF", append_images=frames,
                   save_all=True, duration=30, loop=0)


# --------------- CLASS IMPLEMENTATION --------------- #
class WedgeVortex(object):
    def __init__(self, Nx=520, Ny=180, u_in=0.04, Re=220):
        """
        :param Nx:
        :param Ny:
        :param u_in:
        :param Re:
        """
        # --- input parameters
        self.Nx = Nx  # system size
        self.x_arr = np.arange(0, self.Nx, 1)
        self.Ny = Ny  # system size
        self.y_arr = np.arange(0, self.Ny, 1)
        self.u_in = u_in  # velocity in lattice units
        self.Re = Re  # Reynolds number
        # calculate parameters
        self.nu_LB = self.u_in * self.Ny / (2 * self.Re)  # viscosity
        self.tau = 3 * self.nu_LB + 0.5  # relaxation time

        # --- shape of barrier which we consider
        self.wedge = np.fromfunction(lambda x, y: (abs(x - self.Nx / 4) + abs(y)) < self.Ny / 2,
                                     (self.Nx, self.Ny), dtype=float).T

        self.wall = np.array([x[:] for x in [[False] * self.Nx] * self.Ny])
        self.wall[0, :] = True
        self.wall[self.Ny - 1, :] = True

        # --- initial vectors: directions and weights
        self.e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
        self.W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

        # --- initial density
        rho0 = np.array([x[:] for x in [[1.0] * self.Nx] * self.Ny])
        self.rho = rho0

        # --- initial velocity: u0
        self.epsilon = 0.0001  # velocity perturbation
        vector_zero = [0.0, 0.0]
        u0 = np.array([x[:] for x in [[vector_zero] * self.Nx] * self.Ny])

        for i in range(0, self.Ny):
            val_u0_i = u_in * (1 + self.epsilon * np.sin(self.y_arr[i] / (2 * np.pi * (self.Ny - 1))))
            u0_i = [1 * val_u0_i, 0]
            # update
            u0[i, :] = u0_i

        self.u0 = u0  # velocity vector from outside (streaming), in each point and with components (x,y)
        self.u = np.array([x[:] for x in [[vector_zero] * self.Nx] * self.Ny])  # velocity vector with components (x,y)

        # --- initial distribution at equilibrium
        f_eq = np.array(9 * [[x[:] for x in [[0.0] * self.Nx] * self.Ny]])
        for i in range(0, 9):
            f_eq[i] = self.W[i] * self.rho * (1 + 3 * np.dot(self.u0, self.e[i]) +
                                              9 / 2 * (np.dot(self.u0, self.e[i])) ** 2 +
                                              -3 / 2 * (self.u0[:, :, 0] ** 2 + self.u0[:, :, 1] ** 2))

        self.f_eq = f_eq  # distribution function at equilibrium
        self.f = self.f_eq.copy()  # distribution function
        self.f_col = np.array(9 * [[x[:] for x in [[None] * self.Nx] * self.Ny]])  # collisions distribution function

        # --- printing shapes
        print("rho:", self.rho.shape)
        print("u0:", self.u0.shape)
        print("u:", self.u.shape)
        print("f_eq:", self.f_eq.shape)
        print("f:", self.f.shape)
        print("f_col:", self.f_col.shape)
        print("wedge:", self.wedge.shape)
        print("wall:", self.wall.shape)

        print()
        print("self.wedge")
        pprint(self.wedge)
        print("self.wall.T")
        pprint(self.wall.T)
        print("self.u0")
        pprint(self.u0)

    # --------------------------------------- ALGORITHM: ELEMENTS --------------------------------------- #
    # --- FIRST STEP OF ALGORITHM
    def inlet_flow_first_step(self):
        """
        We will use `f_eq` to start our algorithm. So this function
        deals with inlet flow: you know we interested in static
        flow from outside our box. And this function governs with that.

        :return: Nothing. Just Update the inlet flow.
        -----
        One important remark! Use only in the first step!
        """
        # inlet density: always we will use `u0`
        self.rho[:, 0] = (2 * (self.f_eq[3, :, 0] + self.f_eq[6, :, 0] + self.f_eq[7, :, 0]) +
                          self.f_eq[0, :, 0] + self.f_eq[2, :, 0] + self.f_eq[4, :, 0]) / (1 - np.sqrt(self.u0[:, 0, 0]**2 + self.u0[:, 0, 1]**2))
        # inlet distribution at equilibrium
        for i in range(0, 9):
            self.f_eq[i, :, 0] = self.W[i] * self.rho[:, 0] * (1 + 3 * np.dot(self.u0[:, 0], self.e[i]) +
                                                               9 / 2 * (np.dot(self.u0[:, 0], self.e[i])) ** 2 +
                                                               -3 / 2 * (self.u0[:, 0, 0] ** 2 + self.u0[:, 0, 1] ** 2))

    def inlet_flow(self):
        """
        We will use `f_eq` to start our algorithm. So this function
        deals with inlet flow: you know we interested in static
        flow from outside our box. And this function governs with that.

        :return: Nothing. Just Update the inlet flow.
        -----
        One important remark! Use for every step, except the first step!
        """
        # inlet density: always we will use `u0`
        self.rho[:, 0] = (2 * (self.f[3, :, 0] + self.f[6, :, 0] + self.f[7, :, 0]) +
                          self.f[0, :, 0] + self.f[2, :, 0] + self.f[4, :, 0]) / (1 - np.sqrt(self.u0[:, 0, 0]**2 + self.u0[:, 0, 1]**2))
        # inlet distribution at equilibrium
        for i in range(0, 9):
            self.f_eq[i, :, 0] = self.W[i] * self.rho[:, 0] * (1 + 3 * np.dot(self.u0[:, 0], self.e[i]) +
                                                               9 / 2 * (np.dot(self.u0[:, 0], self.e[i])) ** 2 +
                                                               -3 / 2 * (self.u0[:, 0, 0] ** 2 + self.u0[:, 0, 1] ** 2))

    # --- SECOND STEP OF ALGORITHM
    def apply_bou_con_in_out(self):
        """
        Apply boundary condition for inlet ('in') and outlet ('out') distribution function.
        In this class is important that we have to differentiate the 'normal' distribution
        function and distribution function at equilibrium.

        :return: Nothing. Just update the distribution function for inlet and outlet.
        """
        # inlet distribution function
        self.f[1, :, 0] = self.f_eq[1, :, 0]
        self.f[5, :, 0] = self.f_eq[5, :, 0]
        self.f[8, :, 0] = self.f_eq[8, :, 0]
        # outlet distribution function
        self.f[3, :, self.Nx - 1] = self.f[3, :, self.Nx - 2]
        self.f[6, :, self.Nx - 1] = self.f[6, :, self.Nx - 2]
        self.f[7, :, self.Nx - 1] = self.f[7, :, self.Nx - 2]

    # --- THIRD STEP OF ALGORITHM
    def cal_density_and_distribution(self):
        """
        In the different words, we will calculate the density and distribution
        at equilibrium everywhere - in whole space.

        :return: Nothing. Just update the density and distribution at equilibrium.
        """
        # calculate density
        self.rho = self.f[0] + self.f[1] + self.f[2] + self.f[3] + self.f[4] + \
                   self.f[5] + self.f[6] + self.f[7] + self.f[8]

        # calculate velocity vector
        wec_u = (self.f[0, :, :, np.newaxis] * self.e[0] + self.f[1, :, :, np.newaxis] * self.e[1] +
                 self.f[2, :, :, np.newaxis] * self.e[2] + self.f[3, :, :, np.newaxis] * self.e[3] +
                 self.f[4, :, :, np.newaxis] * self.e[4] + self.f[5, :, :, np.newaxis] * self.e[5] +
                 self.f[6, :, :, np.newaxis] * self.e[6] + self.f[7, :, :, np.newaxis] * self.e[7] +
                 self.f[8, :, :, np.newaxis] * self.e[8])

        self.u[:, :, 0] = wec_u[:, :, 0] / self.rho
        self.u[:, :, 1] = wec_u[:, :, 1] / self.rho

        # print("self.u")
        # pprint(self.u)

        # self.u[:, :, 0] = (self.f[0, :, :] * self.e[0][0] + self.f[1, :, :] * self.e[1][0] +
        #                    self.f[2, :, :] * self.e[2][0] + self.f[3, :, :] * self.e[3][0] +
        #                    self.f[4, :, :] * self.e[4][0] + self.f[5, :, :] * self.e[5][0] +
        #                    self.f[6, :, :] * self.e[6][0] + self.f[7, :, :] * self.e[7][0] +
        #                    self.f[8, :, :] * self.e[8][0]) / self.rho
        #
        # self.u[:, :, 1] = (self.f[0, :, :] * self.e[0][1] + self.f[1, :, :] * self.e[1][1] +
        #                    self.f[2, :, :] * self.e[2][1] + self.f[3, :, :] * self.e[3][1] +
        #                    self.f[4, :, :] * self.e[4][1] + self.f[5, :, :] * self.e[5][1] +
        #                    self.f[6, :, :] * self.e[6][1] + self.f[7, :, :] * self.e[7][1] +
        #                    self.f[8, :, :] * self.e[8][1]) / self.rho

        # distribution at equilibrium in next step
        for i in range(0, 9):
            self.f_eq[i] = self.W[i] * self.rho * (1 + 3 * np.dot(self.u, self.e[i]) +
                                                   9 / 2 * (np.dot(self.u, self.e[i])) ** 2 +
                                                   -3 / 2 * (self.u[:, :, 0] ** 2 + self.u[:, :, 1] ** 2))

    # --- FOURTH STEP OF ALGORITHM
    def cal_collisions_f(self):
        """
        Calculate collisions distributed function.

        :return: Nothing. Just update collisions distributed function.
        """
        for i in range(0, 9):
            self.f_col[i] = self.f[i] - (self.f[i] - self.f_eq[i]) / self.tau

    # --- FIFTH STEP OF ALGORITHM
    def collision_in_obstacle(self):
        """
        If our part of fluid is in the obstacle then we have to consider collisions.
        These collisions change the direction of motion, for example:

            -> change into: <-

        In the other words, we change into opposite direction. We will swap arrays in
        distribution function in this manner (to archive our goal):

            "->" means change into
            0 -> 0
            1 -> 3
            2 -> 4
            3 -> 1
            4 -> 2
            5 -> 7
            6 -> 8
            7 -> 5
            8 -> 6

        :return: Nothing. Just update the directions of parts of fluid.
        -----
        Important remark! We will change: `f_col`
        """
        # prepare revers distribution function
        f_new0 = self.f_col[0]
        f_new1 = self.f_col[3]
        f_new2 = self.f_col[4]
        f_new3 = self.f_col[1]
        f_new4 = self.f_col[2]
        f_new5 = self.f_col[7]
        f_new6 = self.f_col[8]
        f_new7 = self.f_col[5]
        f_new8 = self.f_col[6]

        f_reverse = np.array([f_new0, f_new1, f_new2, f_new3, f_new4, f_new5, f_new6, f_new7, f_new8])

        # if obstacle in obstacle, change direction
        # obstacle = np.logical_or(self.wedge == True, self.wall == True)
        obstacle = self.wedge
        # obstacle = np.array([x[:] for x in [[False] * self.Ny] * self.Nx])

        for i in range(0, 9):
            self.f_col[i] = np.where(obstacle == True, f_reverse[i], self.f_col[i])

    # --- SIXTH STEP OF ALGORITHM
    def streaming_step(self):
        """
        This function hold the streaming step, which means we will move distribution function
        in proper direction. To do this `changing direction` we will use `np.roll` in this
        manner:

            `f_i_j` is corresponding to `f[0, :, :]`
            `f_i_jp1` is corresponding to `f[1, :, :]`
            `f_im1_j` is corresponding to `f[2, :, :]`
            `f_i_jm1` is corresponding to `f[3, :, :]`
            `f_ip1_j` is corresponding to `f[4, :, :]`
            `f_im1_jp1` is corresponding to `f[5, :, :]`
            `f_im1_jm1` is corresponding to `f[6, :, :]`
            `f_ip1_jm1` is corresponding to `f[7, :, :]`
            `f_ip1_jp1` is corresponding to `f[8, :, :]`

        Notation: `f`: distribution function

        :return:
        """
        # upper row
        f_im1_jm1 = np.roll(np.roll(self.f_col[6], -1, axis=0), -1, axis=1)
        f_im1_j = np.roll(np.roll(self.f_col[2], -1, axis=0), 0, axis=1)
        f_im1_jp1 = np.roll(np.roll(self.f_col[5], -1, axis=0), +1, axis=1)
        # same row
        f_i_jm1 = np.roll(np.roll(self.f_col[3], 0, axis=0), -1, axis=1)
        f_i_j = np.roll(np.roll(self.f_col[0], 0, axis=0), 0, axis=1)  # HOW WE ARE!
        f_i_jp1 = np.roll(np.roll(self.f_col[1], 0, axis=0), +1, axis=1)
        # lower row
        f_ip1_jm1 = np.roll(np.roll(self.f_col[7], +1, axis=0), -1, axis=1)
        f_ip1_j = np.roll(np.roll(self.f_col[4], +1, axis=0), 0, axis=1)
        f_ip1_jp1 = np.roll(np.roll(self.f_col[8], +1, axis=0), +1, axis=1)

        self.f = np.array([f_i_j, f_i_jp1, f_im1_j,
                           f_i_jm1, f_ip1_j, f_im1_jp1,
                           f_im1_jm1, f_ip1_jm1, f_ip1_jp1])

    # --------------------------------------- ALGORITHM: DO ONE STEP --------------------------------------- #
    def first_step(self):
        """
        Do one step of evolution of entire system.

        -----
        One important remark! This function deals with first step.
        There exist different function to deal with another steps.
        """
        # Calculate the density and equilibrium distribution on the inlet.
        self.inlet_flow_first_step()
        # Apply the boundary conditions for the inlet and outlet.
        self.apply_bou_con_in_out()
        # Recalculate density and equilibrium distribution everywhere.
        self.cal_density_and_distribution()  # some problem
        # Calculate the collision step.
        self.cal_collisions_f()
        # Replace parts of the distribution function after collision that
        # correspond to fluid-solid boundaries.
        self.collision_in_obstacle()
        # Calculate the streaming step.
        self.streaming_step()

    def do_one_step(self):
        """
        Do one step of evolution of entire system.

        -----
        One important remark! This function deals with all steps,
        beyond first step. There exist different function to deal with
        first step.
        """
        # Calculate the density and equilibrium distribution on the inlet.
        self.inlet_flow()
        # Apply the boundary conditions for the inlet and outlet.
        self.apply_bou_con_in_out()
        # Recalculate density and equilibrium distribution everywhere.
        self.cal_density_and_distribution()  # some problem
        # Calculate the collision step.
        self.cal_collisions_f()
        # Replace parts of the distribution function after collision that
        # correspond to fluid-solid boundaries.
        self.collision_in_obstacle()
        # Calculate the streaming step.
        self.streaming_step()

    # --------------------------------------- CHECKING --------------------------------------- #
    def print_analysis(self):
        print("rho:", self.rho.shape)
        print("u0:", self.u0.shape)
        print("u:", self.u.shape)
        print("f_eq:", self.f_eq.shape)
        print("f:", self.f.shape)
        print("f_col:", self.f_col.shape)

    # --------------------------------------- RETURN DATA --------------------------------------- #
    def return_u0(self):
        return self.u0.copy()

    def return_u(self):
        return self.u.copy()

    def return_u0_proper(self):
        """Do transposition, which is necessary to get better plots."""
        return self.u0.T.copy()

    def return_u_proper(self):
        """Do transposition, which is necessary to get better plots."""
        return self.u.T.copy()

    def return_feq(self):
        return self.f_eq.copy()

    def return_velocity0_magnitude(self):
        # vel_magnitude = np.array([x[:] for x in [[0.0] * self.Ny] * self.Nx])
        vel_magnitude = self.u0[:, :, 0] * self.u0[:, :, 0] + self.u0[:, :, 1] * self.u0[:, :, 1]
        # vel_magnitude[] = (norm(self.u0[:, :])) ** 2

        # for ix in range(0, self.Nx):
        #     for iy in range(0, self.Ny):
        #         vel_magnitude_ixiy = (norm(self.u0[ix, iy])) ** 2
        #         vel_magnitude[ix, iy] = vel_magnitude_ixiy

        return vel_magnitude

    def return_velocity_magnitude(self):
        # vel_magnitude = np.array([x[:] for x in [[0.0] * self.Ny] * self.Nx])
        # for ix in range(0, self.Nx):
        #     for iy in range(0, self.Ny):
        #         vel_magnitude_ixiy = (norm(self.u[ix, iy])) ** 2
        #         vel_magnitude[ix, iy] = vel_magnitude_ixiy

        vel_magnitude = self.u[:, :, 0] * self.u[:, :, 0] + self.u[:, :, 1] * self.u[:, :, 1]

        return vel_magnitude


if __name__ == '__main__':
    Vortex_class = WedgeVortex()
    # velocity at start
    u0_arr = Vortex_class.return_u0()
    u0_x_arr = u0_arr[:, :, 0]
    u0_y_arr = u0_arr[:, :, 1]
    # initial distribution at equilibrium
    init_f = Vortex_class.return_feq()

    # --- Plotting: u0_x: remember you have to do transposition
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"u0_x", fontsize=15, loc='left')
    cax1 = ax.imshow(u0_x_arr, interpolation='nearest', cmap='hot')
    plt.show()

    # --- Plotting: position wedge
    wedge_mat = Vortex_class.wedge

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"wedge", fontsize=15, loc='left')
    cax2 = ax.imshow(wedge_mat, interpolation='nearest', cmap='hot')
    plt.show()

    # --- Plotting: position wall
    wall_mat = Vortex_class.wall

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"wedge", fontsize=15, loc='left')
    cax99 = ax.imshow(wall_mat, interpolation='nearest', cmap='hot')
    plt.show()

    # # --- Plotting: u0_y: remember you have to do transposition
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_title(f"u0_y", fontsize=15, loc='left')
    # cax2 = ax.imshow(u0_y_arr.T, interpolation='nearest', cmap='hot')
    # plt.show()

    # print("Initial destribution at equilibrium")
    # pprint(init_f)
    # print("Shape of init_f:", init_f.shape)

    # --- Plotting: velocity magnitude
    vel0_magnitude = Vortex_class.return_velocity0_magnitude()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"velocity0 magnitude", fontsize=15, loc='left')
    cax3 = ax.imshow(vel0_magnitude, interpolation='nearest', cmap='hot')
    plt.show()

    # ---------------------- DO ONE STEP ---------------------- #
    Vortex_class.first_step()

    # velocity at start
    u0_arr = Vortex_class.return_u0()
    u0_x_arr = u0_arr[:, :, 0]
    u0_y_arr = u0_arr[:, :, 1]

    # --- Plotting: u0_x: remember you have to do transposition
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"u0_x, after one step", fontsize=15, loc='left')
    cax4 = ax.imshow(u0_x_arr, interpolation='nearest', cmap='hot')
    plt.show()

    # --- Plotting: velocity magnitude
    vel_true_magnitude = Vortex_class.return_velocity_magnitude()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"velocity magnitude", fontsize=15, loc='left')
    cax5 = ax.imshow(vel_true_magnitude, interpolation='nearest', cmap='hot')
    plt.show()
