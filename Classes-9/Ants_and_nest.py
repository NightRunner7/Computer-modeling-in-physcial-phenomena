import numpy as np
import random
import sys

# --------------------------- ANTS CLASS --------------------------- #
class AntClass(object):
    def __init__(self, id_ant=None, pos_x=0, pos_y=0):
        """
        :param id_ant: identification of our ant
        :param pos_x: position of ant, `x-axis`. It will be integer: refers to grid column's.
        :param pos_y: position of ant, `y-axis`. It will be integer: refers to grid rows.
        """
        # --- input parameters
        self.id = id_ant
        self.pos_x = pos_x  # position of ant `x`
        self.pos_y = pos_y  # position of ant `y`

        # --- velocity of ant
        self.ex = np.array([1, 0])  # direction of x-axis
        self.ey = np.array([0, 1])  # direction of y-axis
        # possible directors of move
        self.e = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])

        self.n_dirs = len(self.e)  # number of different direction, which ant can take
        self.move_director = None  # the vector of ant's movements
        self.move_id = None  # 'id' to get vector of ant's movement

        # set initial movement
        self.init_random_velocity()

        # --- phase of our ant
        self.food = True  # searching for food
        self.home = False  # coming to home with food

        # --- iterations without being in home and finding food
        self.n_iterations_in_way = 0
        self.beta = 2.0

    # ---------------------------- MOVEMENT OF OUR ANTS ---------------------------- #
    def init_random_velocity(self):
        """Set random direction of ant's movement"""
        move_index = random.randint(1, self.n_dirs) - 1
        # update
        self.move_id = move_index
        self.move_director = self.e[move_index]

    def turn_around(self):
        """Sometimes our ants will turn around"""
        self.move_id = (self.move_id + 4) % self.n_dirs
        self.move_director = self.e[self.move_id]

    def possible_destination(self):
        """
        :return:
        """
        move_ids = [(self.move_id - 1 + self.n_dirs) % self.n_dirs,
                    self.move_id,
                    (self.move_id + 1) % self.n_dirs]

        destination_list = []
        for id in move_ids:
            destination_x = self.pos_x + np.dot(self.e[id], self.ex)
            destination_y = self.pos_y + np.dot(self.e[id], self.ey)
            destination = [destination_x, destination_y]
            # appending
            destination_list.append(destination)

        return destination_list, move_ids

    # ---------------------------- BEHAVIOUR OF ANT ---------------------------- #
    def change_behaviour(self):
        if self.food is True:
            self.food = False
            self.home = True
        else:
            self.food = True
            self.home = False

    def change_movement(self, move_id):
        if move_id >= self.n_dirs:
            sys.exit(f"We have just {self.n_dirs} possible directions. You set id: {move_id},"
                     f" maximum is {self.n_dirs - 1}")

        self.move_id = move_id
        self.move_director = self.e[self.move_id]

    # ---------------------------- LOST ANT ---------------------------- #
    def calculate_spreading_pheromones(self):
        """
        :return:
        """
        # if
        # self.beta = self.beta - 1.0 / 800 * self.n_iterations_in_way

        if self.n_iterations_in_way > 100:
            self.beta = 1.0

        elif self.n_iterations_in_way > 400:
            self.beta = 0.0

        # if self.beta < 0:
        #     self.beta = 0


# --------------------------- SET OF SURROUNDINGS OF NEST --------------------------- #
class SpaceWithAnts(object):
    def __init__(self, grid, nest_loc, nmax_ants=80, alpha=5, const_h=11, beta=0.99):
        """
        :param grid: the grid, which describes the surroundings of the nest.
        :param nest_loc: localization of nest. nest_loc[0]: x-axis, nest_loc[1]: y-axis.
        :param nmax_ants: maximum number of ants in our space.

        There are exist few types of cells:

            `N`: cell, localization of nest.
            `F`: cell, localization of food.
            `O`: cell, localization of open field.
            `B`: cell, localization of obstacle. Ant cannot move into obstacle.
        """
        # --- input parameters
        self.grid = grid
        self.nest_loc = nest_loc
        self.Nx = len(self.grid[0])
        self.Ny = len(self.grid)

        # --- probability and pheromones set up
        self.alpha = alpha  # used in probability
        self.const_h = const_h  # used in probability
        self.beta = beta  # how fast pheromones evaporate

        # --- Food
        self.food_in_nest = 0  # food, which is contains in the nest
        self.food_in_grid = np.where(self.grid == 'F', 100, 0)  # in each square, contains 100 units of food

        # --- Ants
        self.N_ants = 0  # number of ants
        self.Ants = []  # list contain the objects: Ant's.
        self.Nmax_ants = nmax_ants

        # --- grid of pheromones
        self.food_pheromones = np.array([x[:] for x in [[0.0] * self.Ny] * self.Nx])  # leave when searched for food
        self.home_pheromones = np.array([x[:] for x in [[0.0] * self.Ny] * self.Nx])  # leave when went to the nest
        self.all_pheromones = np.array([x[:] for x in [[0.0] * self.Ny] * self.Nx])  # sum of both of them

    # ------------------------------------- PERIODIC BOUNDARY CONDITION ------------------------------------- #
    def per_boundary_con_x(self, pos_x):
        """
        :param pos_x:
        :return:
        """
        return pos_x % self.Nx

    def per_boundary_con_y(self, pos_y):
        """
        :param pos_y:
        :return:
        """
        return pos_y % self.Ny

    # ------------------------------------- SPAWN ANTS ------------------------------------- #
    def spawn_ant(self):
        """
        This function have to spawn a one ant in the nest. Should also
        set the direction of her move, but I implement that in the ant class.
        So in the another word, I already have taken care about it.

        One more thing: ant do first step with random direction!

        :return: Nothing. Just update the data stored in the class.
        """
        # create ant: class object
        ant = AntClass(id_ant=self.N_ants, pos_x=self.nest_loc[0], pos_y=self.nest_loc[1])

        # update ants
        self.Ants.append(ant)
        self.N_ants += 1

        # possible destination
        possible_cells, possible_direction = ant.possible_destination()
        # random direction of her move
        random_direction = random.randint(1, 3) - 1
        move_id = possible_direction[random_direction]
        ant.change_movement(move_id)
        # leave a pheromones
        pos_x_pheromones = ant.pos_x
        pos_y_pheromones = ant.pos_y
        self.food_pheromones[pos_x_pheromones][pos_y_pheromones] += 1
        # change position of ant
        ant.pos_x = self.per_boundary_con_x(possible_cells[random_direction][0])
        ant.pos_y = self.per_boundary_con_y(possible_cells[random_direction][1])

    # ------------------------------------- DEAL WITH PROBABILITY ------------------------------------- #
    def probability_set_direction(self, pheromones_list):
        """
        This function is used to choice the direction when ant
        have pheromones near her.

        :param pheromones_list:
        :return:

        `prob` denotes: probability
        `norm` denotes: normalization
        """
        # --- get normalization probability
        pherm_prob_list = np.array([])
        norm_prob = 0
        for i in range(0, len(pheromones_list)):
            prob = (self.const_h + pheromones_list[i])** self.alpha
            norm_prob += prob
            # appending
            pherm_prob_list = np.append(pherm_prob_list, prob)

        # do normalization
        for i in range(0, len(pherm_prob_list)):
            pherm_prob_list[i] = pherm_prob_list[i] / norm_prob

        # --- get the index, which lead to the direction
        id_direction = np.random.choice(3, 1, replace=False, p=pherm_prob_list)[0]
        return id_direction

    # ------------------------------------- PHEROMONES ------------------------------------- #
    def evaporate_pheromones(self, beta):
        """
        This function describes how fast the pheromones will be evaporate.

        :return: Nothing. Just update the data stored in the class.
        """
        # how quickly the pheromones will evaporate
        self.food_pheromones = self.food_pheromones * beta
        self.home_pheromones = self.home_pheromones * beta

    def lost_ants(self):
        """
        This function will try to deal with lost ants.

        :return:
        """
        for ant_id in range(0, self.N_ants):
            ant_class = self.Ants[ant_id]
            # new iteration
            ant_class.n_iterations_in_way += 1
            # changing the power of spreading pheromones
            ant_class.calculate_spreading_pheromones()

    # ------------------------------------- HOW ANT MOVE ------------------------------------- #
    def move_ants_searching_food(self, ant_id):
        """
        This function will describe the movement of ants, which searching for food
        for her queen.

        :param ant_id: identification of ant, which searching for food.

        :return: Nothing. Just update the data stored in the class.
        """
        # find ant
        ant_class = self.Ants[ant_id]
        # possible destination
        possible_cells, possible_direction = ant_class.possible_destination()

        # --- food in front of ant
        for i in [1, 0, 2]:
            pos_x = self.per_boundary_con_x(possible_cells[i][0])
            pos_y = self.per_boundary_con_y(possible_cells[i][1])
            # check the food is existed
            if self.grid[pos_x][pos_y] == 'F':
                # he/she find something, so he can have strong spreading of pheromones
                ant_class.beta = 2.0
                ant_class.n_iterations_in_way = 0
                # change the velocity direction of ant
                move_id = possible_direction[i]
                ant_class.change_movement(move_id)
                # leave a pheromones
                pos_x_pheromones = ant_class.pos_x
                pos_y_pheromones = ant_class.pos_y
                self.food_pheromones[pos_x_pheromones][pos_y_pheromones] += 1 * ant_class.beta
                # change position of ant
                ant_class.pos_x = pos_x
                ant_class.pos_y = pos_y
                # pick up the food
                self.food_in_grid[pos_x][pos_y] += -1
                # turn around
                ant_class.turn_around()
                # change behaviour
                ant_class.change_behaviour()

                break

        # --- or ant still look for food
        if ant_class.food is True:
            val_pheromones = 0
            pheromones_list = np.array([])
            for i in [0, 1, 2]:
                pos_x = self.per_boundary_con_x(possible_cells[i][0])
                pos_y = self.per_boundary_con_y(possible_cells[i][1])
                val_pheromones += self.home_pheromones[pos_x][pos_y]
                pheromones_list = np.append(pheromones_list, self.home_pheromones[pos_x][pos_y])
            if val_pheromones == 0:
                # none pheromones around, pick up random direction
                random_direction = random.randint(1, 3) - 1
                move_id = possible_direction[random_direction]
                ant_class.change_movement(move_id)
                # leave a pheromones
                pos_x_pheromones = ant_class.pos_x
                pos_y_pheromones = ant_class.pos_y
                self.food_pheromones[pos_x_pheromones][pos_y_pheromones] += 1 * ant_class.beta
                # change position of ant
                ant_class.pos_x = self.per_boundary_con_x(possible_cells[random_direction][0])
                ant_class.pos_y = self.per_boundary_con_y(possible_cells[random_direction][1])
            else:
                # we have some pheromones
                chosen_direction = self.probability_set_direction(pheromones_list)
                move_id = possible_direction[chosen_direction]
                ant_class.change_movement(move_id)
                # leave a pheromones
                pos_x_pheromones = ant_class.pos_x
                pos_y_pheromones = ant_class.pos_y
                self.food_pheromones[pos_x_pheromones][pos_y_pheromones] += 1 * ant_class.beta
                # change position of ant
                ant_class.pos_x = self.per_boundary_con_x(possible_cells[chosen_direction][0])
                ant_class.pos_y = self.per_boundary_con_y(possible_cells[chosen_direction][1])

    def move_ants_going_home(self, ant_id):
        """
        This function will describe the movement of ants, which pick up a food and now
        going back to the nest.

        :param ant_id: identification of ant, which going back to the home.

        :return: Nothing. Just update the data stored in the class.
        """
        # find ant
        ant_class = self.Ants[ant_id]
        # possible destination
        possible_cells, possible_direction = ant_class.possible_destination()

        # --- nest in front of ant
        for i in [1, 0, 2]:
            pos_x = self.per_boundary_con_x(possible_cells[i][0])
            pos_y = self.per_boundary_con_y(possible_cells[i][1])
            # check the food is existed
            if self.grid[pos_x][pos_y] == 'N':
                # he/she find something, so he can have strong spreading of pheromones
                ant_class.beta = 2.0
                ant_class.n_iterations_in_way = 0
                # change the velocity direction of ant
                move_id = possible_direction[i]
                ant_class.change_movement(move_id)
                # leave a pheromones
                pos_x_pheromones = ant_class.pos_x
                pos_y_pheromones = ant_class.pos_y
                self.home_pheromones[pos_x_pheromones][pos_y_pheromones] += 1 * ant_class.beta
                # change position of ant
                ant_class.pos_x = pos_x
                ant_class.pos_y = pos_y
                # pick up the food in the nest
                self.food_in_nest += 1
                # turn around
                ant_class.turn_around()
                # change behaviour
                ant_class.change_behaviour()

                break

        # --- or ant still look for the nest
        if ant_class.home is True:
            val_pheromones = 0
            pheromones_list = np.array([])
            for i in [0, 1, 2]:
                pos_x = self.per_boundary_con_x(possible_cells[i][0])
                pos_y = self.per_boundary_con_y(possible_cells[i][1])
                val_pheromones += self.food_pheromones[pos_x][pos_y]
                pheromones_list = np.append(pheromones_list, self.food_pheromones[pos_x][pos_y])
            if val_pheromones == 0:
                # none pheromones around, pick up random direction
                random_direction = random.randint(1, 3) - 1
                move_id = possible_direction[random_direction]
                ant_class.change_movement(move_id)
                # leave a pheromones
                pos_x_pheromones = ant_class.pos_x
                pos_y_pheromones = ant_class.pos_y
                self.home_pheromones[pos_x_pheromones][pos_y_pheromones] += 1 * ant_class.beta
                # change position of ant
                ant_class.pos_x = self.per_boundary_con_x(possible_cells[random_direction][0])
                ant_class.pos_y = self.per_boundary_con_y(possible_cells[random_direction][1])
            else:
                # we have some pheromones
                chosen_direction = self.probability_set_direction(pheromones_list)
                move_id = possible_direction[chosen_direction]
                ant_class.change_movement(move_id)
                # leave a pheromones
                pos_x_pheromones = ant_class.pos_x
                pos_y_pheromones = ant_class.pos_y
                self.home_pheromones[pos_x_pheromones][pos_y_pheromones] += 1 * ant_class.beta
                # change position of ant
                ant_class.pos_x = self.per_boundary_con_x(possible_cells[chosen_direction][0])
                ant_class.pos_y = self.per_boundary_con_y(possible_cells[chosen_direction][1])

    # ##################### ONE ITERATION ##################### #
    def one_iteration(self):
        # firstly, spawn a one ant
        if self.N_ants <= 80:
            self.spawn_ant()

        # do a move for all of ants
        for ant in range(0, self.N_ants):
            if self.Ants[ant].food is True:
                # move an ant which search food
                self.move_ants_searching_food(ant)
            else:
                # otherwise an ant, going to the home
                self.move_ants_going_home(ant)

        # a pheromones evaporate
        self.evaporate_pheromones(self.beta)

        # dealing with lost ants
        self.lost_ants()

    # --------------------------- RETURN DATA --------------------------- #
    def return_food_pheromones(self):
        return self.food_pheromones.copy()

    def return_home_pheromones(self):
        return self.home_pheromones.copy()

    def return_all_pheromones(self):
        all_pheromones = self.food_pheromones + self.home_pheromones
        return all_pheromones.copy()

    def do_color_coding(self):
        """
        :return:
        """
        cell_N = 1.0 * (np.where(self.grid == 'N', 1, 0)).astype(int)
        cell_O = 2.0 * (np.where(self.grid == 'O', 1, 0)).astype(int)
        cell_F = 3.0 * (np.where(self.grid == 'F', 1, 0)).astype(int)
        cell_B = 4.0 * (np.where(self.grid == 'B', 1, 0)).astype(int)

        state = cell_N + cell_O + cell_F + cell_B
        return state.T.copy()


if __name__ == '__main__':

    # we will check that ids working well: work well
    Ant = AntClass(id_ant=0, pos_x=0, pos_y=0)

    # --- we check how our ants is moving
    for p in range(0, 8):
        Ant.change_movement(p)
        pos_destination, moveId = Ant.possible_destination()
        print("director:", Ant.move_director, "pos_destination:", pos_destination, "moveId:", moveId)

    print()
    # --- we're checking how it will turn around our ants
    for p in range(0, 8):
        Ant.change_movement(p)
        first_director = Ant.move_director
        Ant.turn_around()
        print("director:", first_director, "new_direction:", Ant.move_director, "moveId:", Ant.move_id)

    # print(np.arange(5))