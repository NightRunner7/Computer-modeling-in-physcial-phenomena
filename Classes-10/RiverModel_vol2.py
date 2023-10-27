import numpy as np
import math
from pprint import pprint
import matplotlib.pyplot as plt
import random
import os
import glob
from PIL import Image
# 3D plot
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.colors import LightSource


# ---------------------------------- HELPFUL FUNCTION ---------------------------------- #
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

def get_matrix(_u, _nx, _ny):
    """
    this function deflatten a vector
    so that we can get a nx\times ny matrix
    """
    matrix = np.asarray(_u).reshape((_ny, _nx))
    return matrix

# --------------------------- CLASS IMPLEMENTATION --------------------------- #
class RiverModel(object):
    def __init__(self, Nx=30, Ny=20, const_I=1.0, delta_h=10, beta=0.05, const_r=200):
        """
        :param const_I:
        :param delta_h:
        :param beta:
        :param const_r:
        """
        # --- input parameters
        self.const_I = const_I
        self.delta_h = delta_h
        self.beta = beta
        self.const_r = const_r

        # --- initial grid
        self.Nx = Nx
        self.Ny = Ny
        self.N1D = self.Nx * self.Ny
        self.grid_h = np.zeros(self.N1D)  # 1D array
        self.initial_grid()

        # --- droplet
        # self.droplet_way = np.array([], dtype=int)  # contains the indexes of cells, to mark the droplet movement.
        self.droplet_way = []  # contains the indexes of cells, to mark the droplet movement.

        # --- river
        self.River = np.ones(self.N1D)

    # --------------------------- BASE FUNCTION--------------------------- #
    def initial_grid(self):
        """
        Create the initial grid.

        :return: Nothing. Just update data stored in the class.
        """
        for i in range(0, self.Nx):
            for j in range(0, self.Ny):
                self.grid_h[i + (self.Ny - 1 - j) * self.Nx] = self.const_I * j + 10 ** (-6) * random.random()

        # for i in range(0, self.Nx):
        #     for j in range(0, self.Ny):
        #         self.grid_h[i + (self.Ny - 1 - j) * self.Nx] = self.const_I * j + 0.1 * i

    def apply_bou_con_x(self, index, direction=""):
        """
        Apply boundary condition, what is important we want to set the boundary
        condition only exist `in x`. I have to differentiate the direction of
        our move.

        :param index: reference index to localise the cell.
        :param direction: to differentiate the direction on which I want to move.
        :return: value of the index after applying the boundary condition.

        col: column
        row: row
        """
        if direction == 'left':
            # we are moving to the right
            row = int(index / self.Nx)
            col = index - row * self.Nx
            col_on_left = (col - 1 + self.Nx) % self.Nx

            return row * self.Nx + col_on_left
        elif direction == 'right':
            # we are moving to the right
            row = int(index / self.Nx)
            col = index - row * self.Nx
            col_on_right = (col + 1) % self.Nx

            return row * self.Nx + col_on_right

    # --------------------------- NEIGHBOURS LIST --------------------------- #
    def create_ngb_list(self, index):
        """
        :param index: reference index to localise the cell.

        I introduce the neighbour index. the example grid:

        [0 , 1 , 2 , 3 ]
        [4 , 5 , 6 , 7 ]
        [8 , 9 , 10, 11]
        [12, 13, 14, 15]
        [16, 17, 18, 19]

        Let select cell: `11`. We will have that neighbours:

            loc: 7 , ngb_index: 0 (UPPER)
            loc: 8 , ngb_index: 1 (RIGHT)
            loc: 15, ngb_index: 2 (LOWER)
            loc: 10, ngb_index: 3 (LEFT)

        :return: return list contain localization and indexes of all my neighbours.

        loc: localization
        ngb: neighbour
        """
        if int(index / self.Nx) == 0:
            # --- in this case we do not have UPPER ngb
            # deal with: RIGHT neighbour
            ngb_loc = self.apply_bou_con_x(index, direction='right')  # in which cell, I can find my neighbour
            ngb_index = 1  # index of my neighbour, to differentiate direction to him
            right_ngb = [ngb_loc, ngb_index]

            # deal with: LOWER neighbour
            ngb_loc = index + self.Nx  # in which cell, I can find my neighbour
            ngb_index = 2  # index of my neighbour, to differentiate direction to him
            lower_ngb = [ngb_loc, ngb_index]

            # deal with: LEFT neighbour
            ngb_loc = self.apply_bou_con_x(index, direction='left')  # in which cell, I can find my neighbour
            ngb_index = 3  # index of my neighbour, to differentiate direction to him
            left_ngb = [ngb_loc, ngb_index]

            return [right_ngb, lower_ngb, left_ngb]
        elif int(index / self.Nx) == self.Ny - 1:
            # --- in this case we do not have LOWER ngb
            # deal with: UPPER neighbour
            ngb_loc = index - self.Nx  # in which cell, I can find my neighbour
            ngb_index = 0  # index of my neighbour, to differentiate direction to him
            upper_ngb = [ngb_loc, ngb_index]

            # deal with: RIGHT neighbour
            ngb_loc = self.apply_bou_con_x(index, direction='right')  # in which cell, I can find my neighbour
            ngb_index = 1  # index of my neighbour, to differentiate direction to him
            right_ngb = [ngb_loc, ngb_index]

            # deal with: LEFT neighbour
            ngb_loc = self.apply_bou_con_x(index, direction='left')  # in which cell, I can find my neighbour
            ngb_index = 3  # index of my neighbour, to differentiate direction to him
            left_ngb = [ngb_loc, ngb_index]

            return [upper_ngb, right_ngb, left_ngb]
        else:
            # deal with: UPPER neighbour
            ngb_loc = index - self.Nx  # in which cell, I can find my neighbour
            ngb_index = 0  # index of my neighbour, to differentiate direction to him
            upper_ngb = [ngb_loc, ngb_index]

            # deal with: RIGHT neighbour
            ngb_loc = self.apply_bou_con_x(index, direction='right')  # in which cell, I can find my neighbour
            ngb_index = 1  # index of my neighbour, to differentiate direction to him
            right_ngb = [ngb_loc, ngb_index]

            # deal with: LOWER neighbour
            ngb_loc = index + self.Nx  # in which cell, I can find my neighbour
            ngb_index = 2  # index of my neighbour, to differentiate direction to him
            lower_ngb = [ngb_loc, ngb_index]

            # deal with: LEFT neighbour
            ngb_loc = self.apply_bou_con_x(index, direction='left')  # in which cell, I can find my neighbour
            ngb_index = 3  # index of my neighbour, to differentiate direction to him
            left_ngb = [ngb_loc, ngb_index]

            return [upper_ngb, right_ngb, lower_ngb, left_ngb]

    # --------------------------- IMPLEMENTATION OF AVALANCHES --------------------------- #
    def influence_of_avalanches(self):
        """
        After droplet reaches the depression the avalanches can arrive. We implement
        them to get thicker river's.

        :return: Nothing. Just update the height after avalanches.
        """
        stop_avalanches = False  # condition to find a moment to stop avalanches
        licznik = 0
        while stop_avalanches != True:
            licznik += 1
            stop_avalanches = True
            # print("licznik in avalanches:", licznik)

            for i in range(0, self.N1D):
                # calculate the height difference between neighbours
                ngb_list = self.create_ngb_list(i)
                delta_h_ngb = []
                for j in range(0, len(ngb_list)):
                    ngb_index = ngb_list[j][0]
                    delta_h_cell = self.grid_h[i] - self.grid_h[ngb_index]
                    # appending
                    delta_h_ngb.append(delta_h_cell)

                # influences of avalanches
                if max(delta_h_ngb) > self.const_r:
                    stop_avalanches = False
                    self.grid_h[i] = self.grid_h[i] - 0.25 * self.delta_h

    def influence_of_avalanches_vol2(self):
        """
        After droplet reaches the depression the avalanches can arrive. We implement
        them to get thicker river's.

        :return: Nothing. Just update the height after avalanches.
        -------
        Important remark: THIS VERSION IS MUCH FASTER!
        """
        # change self.grid into 2D array
        grid_h_2D = np.asarray(self.grid_h).reshape(self.Ny, self.Nx)
        stop_avalanches = False  # condition to find a moment to stop avalanches
        # licznik = 0

        while stop_avalanches != True:
            stop_avalanches = True
            # licznik += 1
            # print("licznik", licznik)

            # --- find neighbours
            # upper row
            g_im1_j = np.roll(np.roll(grid_h_2D, -1, axis=0), 0, axis=1)
            g_im1_j[-1, :] = grid_h_2D[-1, :]
            # same row
            g_i_jm1 = np.roll(np.roll(grid_h_2D, 0, axis=0), -1, axis=1)
            g_i_jp1 = np.roll(np.roll(grid_h_2D, 0, axis=0), +1, axis=1)
            # lower row
            g_ip1_j = np.roll(np.roll(grid_h_2D, +1, axis=0), 0, axis=1)
            g_ip1_j[0, :] = grid_h_2D[0, :]  # we do not have boundary condition in y-axis

            ngb_2D = [g_im1_j, g_i_jm1, g_i_jp1, g_ip1_j]

            # --- calculate differences
            delta_h_2D = []
            for i in range(0, 4):
                delta_h_i_2D = grid_h_2D - ngb_2D[i]
                # appending
                delta_h_2D.append(delta_h_i_2D)

            # --- calculate the height difference between neighbours
            # Combine all arrays into a 3D array
            combined_delta_h = np.stack((delta_h_2D[0], delta_h_2D[1], delta_h_2D[2], delta_h_2D[3]))
            # Find the maximum value at each index along the axis 0
            max_delta_h = np.max(combined_delta_h, axis=0)

            # --- influences of avalanches
            after_avalanch = np.where(max_delta_h > self.const_r, grid_h_2D - 0.25 * self.delta_h, grid_h_2D)
            if np.array_equal(after_avalanch, grid_h_2D) == False:
                stop_avalanches = False

            # --- update value of grid_h_2D
            grid_h_2D = after_avalanch

        # --- after while: flatt `grid_h_2D` and update grid_h
        self.grid_h = grid_h_2D.flatten()

    # --------------------------- IMPLEMENTED ONE DROPLET PATH --------------------------- #
    def put_droplet(self):
        """
        This function will add one droplet to the list. Firstly we have to
        randomise the localization of droplet.

        :return: Nothing just
        """
        droplet_index = random.randint(1, self.N1D) - 1
        # appending to the list
        # self.droplet_way = np.append(self.droplet_way, droplet_index)
        self.droplet_way.append(droplet_index)

    def choose_next_cell(self, ngb_list):
        """
        Choose the next localization depending on neighbours of the
        last appearance of the droplet.

        :param ngb_list: list containg all of neigbours of the last localization
            of the droplet.
        :return: index of the next location of the droplet.
        """
        droplet_index = self.droplet_way[-1]  # take last localization of droplet
        ngb_indexes = np.array([])

        # --- calculate w-matrix
        w_list = np.array([])
        for i in range(len(ngb_list)):
            ngb_index = ngb_list[i][0]  # localization of one of my neighbour
            hi_minus_hj = self.grid_h[droplet_index] - self.grid_h[ngb_index]
            wi = np.heaviside(hi_minus_hj, 1) * np.exp(self.beta * hi_minus_hj)
            # appending
            ngb_indexes = np.append(ngb_indexes, ngb_index)
            w_list = np.append(w_list, wi)

        # --- finding: sum of `wi`
        w_sum = sum(w_list)

        # --- finding the probability: P_ij
        Pij_list = np.array([])
        for i in range(len(ngb_list)):
            P_i = w_list[i] / w_sum
            # appending
            Pij_list = np.append(Pij_list, P_i)

        # --- choose neighbour
        droplet_new_loc = np.random.choice(ngb_indexes, 1, replace=False, p=Pij_list)[0]

        return int(droplet_new_loc)

    def one_droplet_till_end(self):
        """
        Governs path of one droplet. This function is responsible for eroding
        proces, which occurs in whole path of the droplet.

        :return: Nothing. Just update the height grid after droplet path.
        """
        # --- in the beginning we have to set initial location of droplet
        self.put_droplet()

        # --- find a way to the depression
        while int(self.droplet_way[-1] / self.Nx) < self.Ny - 1:
            # --- we will continue journey of droplet till reaches the depression
            # find neighbours of last localization of droplet
            ngb_list = self.create_ngb_list(self.droplet_way[-1])
            # choose next localization
            next_localization = self.choose_next_cell(ngb_list)
            # update localization
            # self.droplet_way = np.append(self.droplet_way, next_localization)
            self.droplet_way.append(next_localization)

        # --- erode whole path of droplet
        path_no_repetition = list(set(self.droplet_way))  # during the trip we want to erode just once.

        for i in range(0, len(path_no_repetition)):
            index = path_no_repetition[i]
            self.grid_h[index] = self.grid_h[index] - self.delta_h

        # --- in the end I have to erase path of droplet
        # self.droplet_way = np.array([], dtype=int)
        self.droplet_way = []

        # --- include influence of avalanches
        # self.influence_of_avalanches()
        self.influence_of_avalanches_vol2()

    def one_droplet_till_end_vol2(self):
        """
        Governs path of one droplet. This function is responsible for eroding
        proces, which occurs in whole path of the droplet.

        :return: Nothing. Just update the height grid after droplet path.
        ------
        Important remark: version without avalanches
        """
        # --- in the beginning we have to set initial location of droplet
        self.put_droplet()

        # --- find a way to the depression
        while int(self.droplet_way[-1] / self.Nx) < self.Ny - 1:
            # --- we will continue journey of droplet till reaches the depression
            # find neighbours of last localization of droplet
            ngb_list = self.create_ngb_list(self.droplet_way[-1])
            # choose next localization
            next_localization = self.choose_next_cell(ngb_list)
            # update localization
            # self.droplet_way = np.append(self.droplet_way, next_localization)
            self.droplet_way.append(next_localization)

        # --- erode whole path of droplet
        path_no_repetition = list(set(self.droplet_way))  # during the trip we want to erode just once.

        for i in range(0, len(path_no_repetition)):
            index = path_no_repetition[i]
            self.grid_h[index] = self.grid_h[index] - self.delta_h

        # --- in the end I have to erase path of droplet
        # self.droplet_way = np.array([], dtype=int)
        self.droplet_way = []

    # --------------------------- IMPLEMENTED NUMEROUS OF DROPLET's PATHS --------------------------- #
    def path_of_river(self):
        """:return: Nothing. Just present how should the river look like."""
        for i in range(0, self.N1D):
            # for all cells in the grid
            # self.River[i] += 1  # I visit that cell
            river_cell = i

            while int(river_cell / self.Nx) < self.Ny - 1:
                ngb_list = self.create_ngb_list(river_cell)
                height_list = []
                for j in range(0, len(ngb_list)):
                    ngb_index = ngb_list[j][0]
                    cell_height = self.grid_h[ngb_index]
                    # appending
                    height_list.append(cell_height)

                min_height = min(height_list)
                min_index = height_list.index(min_height)
                # where I should go
                river_cell = ngb_list[min_index][0]
                self.River[river_cell] += 1  # I visit that cell

    def droplets_till_end(self, n_droplet=1, part_to_river=100):
        """
        :param n_droplet: number of droplets we want to consider.
        :return: Nothing. Just consider `n_droplet` droplets.
        """
        for i in range(0, n_droplet):
            print("-------------------------------------------------")
            print("Which droplet?:", i)
            self.one_droplet_till_end()

            if (i + 172) % part_to_river == 0:
                self.path_of_river()

    # --------------------------- RETURN DATA --------------------------- #
    def return_grid_h(self):
        """:return: the reshaped grid of height."""
        return np.asarray(self.grid_h).reshape((self.Ny, self.Nx))

    def return_river_grid(self, threshold=0):
        """:return: the reshaped grid of river."""
        # self.River[:] -= threshold  # threshold
        # self.River[:] = self.River[:] * (-1)

        map_threshold = np.where(self.River > threshold, 1, 0)
        result = np.asarray(map_threshold).reshape((self.Ny, self.Nx))

        pprint(result)
        pprint(self.River)

        return result

if __name__ == '__main__':
    Nx_test = 60  # 60
    Ny_test = 40  # 40
    n_droplet_test = 10000  # 100000
    simple_threshold = int(n_droplet_test / 100)
    # --- create and test classes
    RiverClass = RiverModel(Nx=Nx_test, Ny=Ny_test)
    print("RiverClass.Nx:", RiverClass.Nx, "RiverClass.Ny", RiverClass.Ny)

    # --- Elevation map after: 0 droplet
    h0 = RiverClass.return_grid_h()
    matrix_h0 = get_matrix(h0, Nx_test, Ny_test)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"Elevation map after: 0 droplet", fontsize=15, loc='left')
    cax1 = plt.imshow(matrix_h0, interpolation='nearest', cmap='jet')
    plt.colorbar(cax1)
    plt.show()

    # --- Elevation map after: 1 droplet
    RiverClass.one_droplet_till_end()
    h0 = RiverClass.return_grid_h()
    matrix_h0 = get_matrix(h0, Nx_test, Ny_test)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"Elevation map after: 1 droplet", fontsize=15, loc='left')
    cax2 = ax.imshow(matrix_h0, interpolation='nearest', cmap='jet')
    plt.colorbar(cax2)
    plt.show()

    # --- Elevation map after: `n_droplet_test + 1` droplet
    RiverClass.droplets_till_end(n_droplet_test)
    river_h0 = RiverClass.return_river_grid(threshold=simple_threshold)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"Elevation map after: {n_droplet_test+1} droplet", fontsize=15, loc='left')
    cax3 = ax.imshow(river_h0, interpolation='nearest', cmap='jet')
    plt.colorbar(cax3)
    plt.show()

    # --- PLOT 3D
    # topography = RiverClass.return_grid_h()
    # fig = go.Figure(data=[go.Surface(z=topography)])
    # # fig.update_layout(title='Mt Bruno Elevation', autosize=False,
    # #                   width=500, height=500,
    # #                   margin=dict(l=65, r=50, b=65, t=90))
    # fig.update_layout(title='Area topography', autosize=True)
    # fig.show()

    # Load and format data
    topography = RiverClass.return_grid_h()
    river_map = RiverClass.return_river_grid(threshold=simple_threshold)

    z = topography
    nrows, ncols = z.shape
    x = np.linspace(0, Nx_test, ncols)
    y = np.linspace(0, Ny_test, nrows)
    x, y = np.meshgrid(x, y)

    # region = np.s_[5:50, 5:50]
    # x, y, z = x[region], y[region], z[region]

    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    # cm.gist_earth
    rgb = ls.shade(river_map, cmap=cm.coolwarm, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)
    # plt.colorbar(rgb)
    plt.show()
