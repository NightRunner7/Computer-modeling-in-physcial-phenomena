import numpy as np
import random
import sys
from pprint import pprint


# --------------- HELPFUL FUNCTION --------------- #
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

# --------------------------- CLASS FISH AND SHARKS --------------------------- #
class FishClass(object):
    def __init__(self, id_fish=None, pos_x=0, pos_y=0, fish_rep=3):
        """
        :param id_fish: identification of our fish.
            (int)
        :param pos_x: position of fish, `x-axis`. It will be integer: refers to grid column's.
            (int)
        :param pos_y: position of fish, `y-axis`. It will be integer: refers to grid rows.
            (int)
        :param fish_rep: reproduction speed of the fish.
            (int)
        """
        # --- input parameters
        self.id = id_fish
        self.pos_x = pos_x  # position of fish `x`
        self.pos_y = pos_y  # position of fish `y`
        self.fish_rep = random.randint(0, fish_rep)

    def update_position(self, _pos_x, _pos_y):
        """
        :param _pos_x: new position of fish, `x-axis`. It will be integer: refers to grid column's.
            (int)
        :param _pos_y: new position of fish, `y-axis`. It will be integer: refers to grid rows.
            (int)
        :return: Nothing just update the position of the shark.
        """
        self.pos_x = _pos_x
        self.pos_y = _pos_y


class SharkClass(object):
    def __init__(self, id_shark=None, pos_x=0, pos_y=0, hungry_lvl=3, shark_rep=20):
        """
        :param id_shark: identification of our fish.
            (int)
        :param pos_x: position of shark, `x-axis`. It will be integer: refers to grid column's.
            (int)
        :param pos_y: position of shark, `y-axis`. It will be integer: refers to grid rows.
            (int)
        :param hungry_lvl: hungry level of shark's. Each steps, which shark didn't
            eat a fish it drops by one. If it goes to zero, shark will die.
            (int)
        :param shark_rep: reproduction speed of the shark.
            (int)
        """
        # --- input parameters
        self.id = id_shark
        self.pos_x = pos_x  # position of fish `x`
        self.pos_y = pos_y  # position of fish `y`
        self.init_hungry_lvl = hungry_lvl  # initial hungry level
        self.hungry_lvl = hungry_lvl
        # self.shark_rep = shark_rep
        self.shark_rep = random.randint(0, shark_rep)

        # --- I ate a fish
        self.found_food = False

    def eating_process(self):
        """
        This function governs the eating fish of the shark.
        :return: Nothing just update the data stored in class.
        """
        if self.found_food is True:
            self.hungry_lvl = self.init_hungry_lvl
            self.found_food = False  # because I have already eaten a fish
        else:
            self.hungry_lvl -= 1

    def update_position(self, _pos_x, _pos_y):
        """
        :param _pos_x: new position of shark, `x-axis`. It will be integer: refers to grid column's.
            (int)
        :param _pos_y: new position of shark, `y-axis`. It will be integer: refers to grid rows.
            (int)
        :return: Nothing just update the position of the shark.
        """
        self.pos_x = _pos_x
        self.pos_y = _pos_y

# --------------------------- MAIN CLASS --------------------------- #
class WaTorUniverse(object):
    def __init__(self, n_size=40, n_fish=300, n_shark=10, fish_rep=3, shark_rep=20, hungry_lvl=3):
        """
        :param n_size: initial size of our grid: 2D grid (NxN).
            (int)
        :param n_fish: initial number of fish.
            (int)
        :param n_shark: initial number of shark's.
            (int)
        :param fish_rep: time steps to event of fish reproduction.
            (int)
        :param shark_rep: time steps to event of shark reproduction.
            (int)
        :param hungry_lvl: hungry level of shark's. Each steps, which shark didn't
            eat a fish it drops by one. If it goes to zero, shark will die.
            (int)
        """
        # --- input parameters
        self.n_fish = n_fish
        self.n_shark = n_shark
        self.fish_rep = fish_rep
        self.shark_rep = shark_rep
        self.init_hungry_lvl = hungry_lvl

        # --- id of the fish and sharks
        self.free_id_fish = np.arange(1, n_size * n_size, 1, dtype=int)
        self.free_id_shark = np.arange(-1, (-1) * n_size * n_size, -1, dtype=int)

        # --- list of fish and sharks
        self.shark_dict = {}
        self.fish_dict = {}

        # --- lattice or grid of our system
        self.n_size = n_size
        self.system_grid = self.create_initial_grid()

        # possible directions
        self.e = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])

    # ------------------------------------- SPAWN FISH AND SHARK's ------------------------------------- #
    def create_initial_grid(self):
        """
        This class has to create initial grid of our system. We will use notation as follows:

            'W': cell, which is empty.
            'F': cell, which is contains the fish.
            'S': cell, which iis contains the shark.

        :return: the initial grid of iur system.
        -----
        Important remark! Each cell can contain only one fish or one shark.
        """
        init_grid = np.array([[None] * self.n_size] * self.n_size)
        fish_and_sharks = self.n_fish + self.n_shark
        # randomise a grid with sharks and fish.
        flat_indexes_defectors = np.random.choice(self.n_size * self.n_size, fish_and_sharks, replace=False)

        for fish in range(0, self.n_fish):
            index = flat_indexes_defectors[fish]
            index_i, index_j = get_index_2d(index, self.n_size)
            # create a class object
            id_fish = self.free_id_fish[0]
            self.free_id_fish = self.free_id_fish[1:]
            one_fish = FishClass(id_fish=id_fish, pos_x=index_i, pos_y=index_j, fish_rep=self.fish_rep)
            # append a class object
            init_grid[index_i][index_j] = {'type': 'F', 'class': one_fish}
            self.fish_dict[id_fish] = one_fish

        for shark in range(0, self.n_shark):
            index = flat_indexes_defectors[shark + self.n_fish]
            index_i, index_j = get_index_2d(index, self.n_size)
            # create a class object
            id_shark = self.free_id_shark[0]
            self.free_id_shark = self.free_id_shark[1:]
            one_shark = SharkClass(id_shark=id_shark, pos_x=index_i, pos_y=index_j,
                                   hungry_lvl=self.init_hungry_lvl, shark_rep=self.shark_rep)
            # append a class object
            init_grid[index_i][index_j] = {'type': 'S', 'class': one_shark}
            self.shark_dict[id_shark] = one_shark

        # set empty cell: water
        init_grid = np.where(init_grid == None, {'type': 'W', 'class': None}, init_grid)

        return init_grid

    # ------------------------------------- BASIC AND NGB LIST ------------------------------------- #
    def apply_per_bou_con(self, position):
        """
        Apply periodic boundary condition. One condition we consider
        only square grid.

        :param position: one of the position `x-axis` or `y-axis`.
            (int)
        :return: position after applied periodic boundary condition.
        """
        return (position + self.n_size) % self.n_size

    def create_ngb_list(self, pos_x, pos_y):
        """
        Create a neighborhood list to the selected position x and y
        :return: the list of my neighbours.
        """
        # ngb_class_list = []
        ngb_type_list = []
        for i in range(len(self.e)):
            ngb_pos_x = self.apply_per_bou_con(pos_x + self.e[i][0])
            ngb_pos_y = self.apply_per_bou_con(pos_y + self.e[i][1])
            ngb_cell = self.system_grid[ngb_pos_x][ngb_pos_y]
            ngb_type = ngb_cell['type']
            # ngb_class = ngb_cell['class']
            # appending
            ngb_type_list.append(ngb_type)
            # ngb_class_list.append(ngb_class)

        return ngb_type_list

    # ------------------------------------- HELPFUL FUNCTION TO REPRODUCE  ------------------------------------- #
    def fish_reproduction(self, _new_fish_pos_x, _new_fish_pos_y, time_to_reproduction):
        """
        :param _new_fish_pos_x: position `x-axis` of the new fish. (int)
        :param _new_fish_pos_y: position `y-axis` of the new fish. (int)
        :param time_to_reproduction: variable, which describe is the fish will reproduce. (int)
        :return: Nothing. Just put a new fish in new place (if the reproduction occurred)
        ------
        One important remark! May happen that value of the `time_to_reproduction` will be negative. Still that means
        if fish can reproduce, it will reproduce.
        """
        if time_to_reproduction > 0:
            # No reproduce. Old fish position: water
            self.system_grid[_new_fish_pos_x][_new_fish_pos_y] = {'type': 'W', 'class': None}
        else:
            # Time to reproduce. Old fish position: new fish
            new_id_fish = self.free_id_fish[0]
            self.free_id_fish = self.free_id_fish[1:]
            new_fish = FishClass(id_fish=new_id_fish, pos_x=_new_fish_pos_x, pos_y=_new_fish_pos_y,
                                 fish_rep=self.fish_rep)
            # appending new fish
            self.system_grid[_new_fish_pos_x][_new_fish_pos_y] = {'type': 'F', 'class': new_fish}
            self.fish_dict[new_id_fish] = new_fish
            self.n_fish += 1

    def shark_reproduction(self, _new_shark_pos_x, _new_shark_pos_y, time_to_reproduction):
        """
        :param _new_shark_pos_x: position `x-axis` of the new shark. (int)
        :param _new_shark_pos_y: position `y-axis` of the new shark. (int)
        :param time_to_reproduction: variable, which describe is the shark will reproduce. (int)
        :return: Nothing. Just put a new shark in new place (if the reproduction occurred). (int)
        ------
        One important remark! May happen that value of the `time_to_reproduction` will be negative. Still that means
        if shark can reproduce, it will reproduce.
        """
        if time_to_reproduction > 0:
            # old shark position: water
            self.system_grid[_new_shark_pos_x][_new_shark_pos_y] = {'type': 'W', 'class': None}
        else:
            # old shark position: new shark
            new_id_shark = self.free_id_shark[0]
            self.free_id_shark = self.free_id_shark[1:]
            new_shark = SharkClass(id_shark=new_id_shark, pos_x=_new_shark_pos_x, pos_y=_new_shark_pos_y,
                                   hungry_lvl=self.init_hungry_lvl, shark_rep=self.shark_rep)
            # appending new shark
            self.system_grid[_new_shark_pos_x][_new_shark_pos_y] = {'type': 'S', 'class': new_shark}
            self.shark_dict[new_id_shark] = new_shark
            self.n_shark += 1

    # ------------------------------------- MOVING AND REPRODUCTION: SPECIES ------------------------------------- #
    def fish_move_and_reproduction(self):
        """
        This function deals with motion of the fish and their reproduction.

        :return: Nothing just update the grid.
        """
        # --- fin all the keys of the fish
        fish_id_list = list(self.fish_dict.keys())

        for i in range(0, len(fish_id_list)):
            # --- consider move of the fish
            fish_id = fish_id_list[i]
            one_fish = self.fish_dict[fish_id]
            old_pos_x = one_fish.pos_x
            old_pos_y = one_fish.pos_y
            # ngb list
            ngb_type_list = self.create_ngb_list(old_pos_x, old_pos_y)
            # Find indices of all occurrences of "W" in neighbourhood list
            water_to_move = [index for index, value in enumerate(ngb_type_list) if value == 'W']
            if water_to_move:
                # --- the fish will move into new cell: water
                which_ngb = random.choice(water_to_move)  # Select a random index from neighbours
                new_pos_x = self.apply_per_bou_con(one_fish.pos_x + self.e[which_ngb][0])
                new_pos_y = self.apply_per_bou_con(one_fish.pos_y + self.e[which_ngb][1])
                one_fish.update_position(new_pos_x, new_pos_y)
                # --- taking into account reproduction or not
                self.system_grid[new_pos_x][new_pos_y] = {'type': 'F', 'class': one_fish}  # new fish position
                time_to_rep = one_fish.fish_rep  # time to reproduce
                self.fish_reproduction(old_pos_x, old_pos_y, time_to_rep)

    def shark_move_and_reproduction(self):
        """
        This function deals with motion of the shark and their reproduction.

        :return: Nothing just update the grid.
        """
        # --- fin all the keys of the fish
        shark_id_list = list(self.shark_dict.keys())

        for i in range(0, len(shark_id_list)):
            # --- consider a move of a shark
            shark_id = shark_id_list[i]
            one_shark = self.shark_dict[shark_id]
            old_pos_x = one_shark.pos_x
            old_pos_y = one_shark.pos_y
            # ngb list
            ngb_type_list = self.create_ngb_list(old_pos_x, old_pos_y)
            # Find indices of all occurrences of "W" in neighbourhood list
            fish_to_eat = [index for index, value in enumerate(ngb_type_list) if value == 'F']
            water_to_move = [index for index, value in enumerate(ngb_type_list) if value == 'W']
            if fish_to_eat:
                # --- shark will eat one of the fish
                which_ngb = random.choice(fish_to_eat)  # Select a random index from neighbours
                new_pos_x = self.apply_per_bou_con(one_shark.pos_x + self.e[which_ngb][0])
                new_pos_y = self.apply_per_bou_con(one_shark.pos_y + self.e[which_ngb][1])
                one_shark.update_position(new_pos_x, new_pos_y)
                # --- update date of killing fish
                one_shark.found_food = True  # found food
                self.n_fish -= 1  # decrease a fish by one
                index_fish_to_kill = self.system_grid[new_pos_x][new_pos_y]['class'].id
                self.free_id_fish = np.append(self.free_id_fish, index_fish_to_kill)
                del self.fish_dict[index_fish_to_kill]  # delete a fish, which has been eaten by shark
                # --- taking into account reproduction or not
                self.system_grid[new_pos_x][new_pos_y] = {'type': 'S', 'class': one_shark}  # new shark position
                time_to_rep = one_shark.shark_rep  # time to reproduce
                self.shark_reproduction(old_pos_x, old_pos_y, time_to_rep)
            elif water_to_move:
                # --- shark will move into new cell: water
                which_ngb = random.choice(water_to_move)  # Select a random index from neighbours
                new_pos_x = self.apply_per_bou_con(one_shark.pos_x + self.e[which_ngb][0])
                new_pos_y = self.apply_per_bou_con(one_shark.pos_y + self.e[which_ngb][1])
                one_shark.update_position(new_pos_x, new_pos_y)
                # --- taking into account reproduction or not
                self.system_grid[new_pos_x][new_pos_y] = {'type': 'S', 'class': one_shark}  # new fish position
                time_to_rep = one_shark.shark_rep  # time to reproduce
                self.shark_reproduction(old_pos_x, old_pos_y, time_to_rep)

    # ------------------------------------- UPDATE THE REPRODUCTION TIME ------------------------------------- #
    def update_reproduction_time(self):
        """
        This function will hold the method of time to reproduce. After one time step all
        the variables, which describe time to reproduce: `shark_rep` and `fish_rep` will drop by one.

        :return: Nothing. Just update the data stored in the class.
        ------
        One important remark! It may happen that one fish will have `fish_rep = 0` and will not reproduce
        in the next step, because can be surrounded by other fishes - so will cannot move. That is the case
        we will take `<= 0` during reproduction step - in the other function.
        """
        # --- find all the keys of the fish
        shark_id_list = list(self.shark_dict.keys())

        for i in range(0, len(shark_id_list)):
            shark_id = shark_id_list[i]
            # drop by one the time to reproduction
            self.shark_dict[shark_id].shark_rep -= 1

        # --- fin all the keys of the fish
        fish_id_list = list(self.fish_dict.keys())

        for i in range(0, len(fish_id_list)):
            fish_id = fish_id_list[i]
            # drop by one the time to reproduction
            self.fish_dict[fish_id].fish_rep -= 1

    # ------------------------------------- HUNGRY SHARKS ------------------------------------- #
    def hungry_level_sharks(self):
        """
        This function will show how to deal with the fact that sharks will get hungry
        each time step without eating the fish. If some shark deos not ate any fish during
        few time steps, it will die.

        :return: Nothing. Just update the data stored in class.
        """
        # --- fin all the keys of the fish
        shark_id_list = list(self.shark_dict.keys())

        for i in range(0, len(shark_id_list)):
            shark_id = shark_id_list[i]  # id of considering shark
            one_shark = self.shark_dict[shark_id]
            # --- get hungry
            one_shark.eating_process()

            # --- shark will die
            if one_shark.hungry_lvl == 0:
                pos_x = one_shark.pos_x
                pos_y = one_shark.pos_y
                # update data
                self.n_shark -= 1
                self.system_grid[pos_x][pos_y] = {'type': 'W', 'class': None}  # update the grid
                del self.shark_dict[shark_id]  # update the dictionary with sharks
                # taking care of the shark index
                self.free_id_shark = np.append(self.free_id_shark, shark_id)

    # --------------------------------------- RETURN DATA --------------------------------------- #
    def return_grid_type(self):
        grid_type = np.array([[None] * self.n_size] * self.n_size)
        for ix in range(0, self.n_size):
            for iy in range(0, self.n_size):
                type = self.system_grid[ix][iy]['type']
                grid_type[ix][iy] = type

        return grid_type

    def do_color_coding(self):
        """
        :return:
        """
        grid_type = self.return_grid_type()

        cell_N = 0.0 * (np.where(grid_type == 'W', 1, 0)).astype(int)
        cell_O = 1.0 * (np.where(grid_type == 'F', 1, 0)).astype(int)
        cell_F = 2.0 * (np.where(grid_type == 'S', 1, 0)).astype(int)

        state = cell_N + cell_O + cell_F
        return state.T.copy()

if __name__ == '__main__':
    print("I m here")

    N_size = 10
    N_fish = 80
    N_shark = 10
    Fish_rep = 3
    Shark_rep = 20
    Hungry_lvl = 3

    SystemClass = WaTorUniverse(n_size=N_size, n_fish=N_fish, n_shark=N_shark,
                                fish_rep=Fish_rep, shark_rep=Shark_rep, hungry_lvl=Hungry_lvl)

    print("The initial grid of our system")
    pprint(SystemClass.return_grid_type())

    OneShark = SharkClass()
    OneShark.eating_process()
    print("the hungry level of the shark:", OneShark.hungry_lvl)

    OneShark.found_food = True
    OneShark.eating_process()
    print("the hungry level of the shark:", OneShark.hungry_lvl)
