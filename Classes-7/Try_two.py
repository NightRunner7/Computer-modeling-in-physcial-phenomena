import numpy as np
from pprint import pprint
# import sys
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

def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.PNG")]
    frame_one = frames[0]
    frame_one.save(f"{frame_folder}/Evolution-prison-dilemma-movie.gif", format="GIF", append_images=frames,
                   save_all=True, duration=30, loop=0)

# --------------- CLASS IMPLEMENTATION --------------- #
class PrisonDilemma(object):
    """
    This class will contain methods to dealing with Prison dilemma in the grid. That means
    we have a grid, where in each cell we set a `player` which can cooperate or defect.
    Each player will be play a one game with all his neighbourhood and with himself.
    In the other words, in one step time he will have nine games.

    One important remark: each player in fixed step time, have fixed strategy. So if
    the player strategy is `cooperate` he will be cooperated with all the games.

    To differentiate players with different strategy we will use the following manner:

        `Cop`: describes the cooperative player (or with cooperative strategy) - always cooperative
        `Def`: describes the defection player (or with defection strategy) - always defecting
        `Pav`: describes the pavlov player (or with pavlov strategy) - at firsts cooperates, then
        changing his strategy each time if fixed player changed himself strategy.
        `Tit`: describes the player with strategy: `Tit-for-tap` - at first cooperates, then
        adapts his strategy to previous strategy of his opponent (neighbour).

    Thus, that means our gird will have in each cell: `Cop`, `Def`, `Pav` or `Tit`.
    ---------
    Also I want to point out that in each game only two `behaviours` are possible:

        `C`: denotes being cooperative
        `D`: denotes being deflecting
    """

    def __init__(self, initial_grid, val_b=1.0, m_games=5):
        """
        :param initial_grid: lattice or grid, which describes initial condition in our case.
            That means it is an array, which has elements: `Cop` or `Def`, `Pav` or `Tit`, to
            differentiate the strategy of player, localised in cell.
            (array: looks like square matrix)
        :param val_b: the parameter, which will change. This parameter describes the `payoff`
            or reward on betray someone of his neighbours.
            (float)

        str: strategy
        beh: behaviour
        """
        self.m_games = m_games  # how many games we will do before changing strategy
        # --- Strategy and behaviour
        self.str_grid = initial_grid
        self.n_size = len(initial_grid)
        # how will look like initial behaviour of our players.
        # In the beginning only completely deflectors (`Def`) will be deflecting.
        self.n_ngb = 9
        initial_beh = np.where(self.str_grid == 'Def', 'D', 'C')
        self.beh_grid = np.array([initial_beh] * self.n_ngb)

        # --- payoffs: describes the results or reward of one game
        self.payoff_cc = 1  # I being cooperative and my opponent being cooperative
        self.payoff_cd = 0  # I being cooperative, but my opponent defected me
        self.payoff_dc = val_b  # I being deflated and my opponent being cooperative
        self.payoff_dd = 0  # I being deflated as my opponent
        self.payoff_grid = np.array([[0.0] * self.n_size] * self.n_size)

        # --- My neighbours `behaviour`
        self.beh_ngb_now = self.create_beh_ngb_now()

    # --------------------------------------- BEHAVIOUR OF MY NEIGHBOURS --------------------------------------- #
    def create_beh_ngb_now(self):
        """
        :return: Nothing. Just stored information, how looks my neighbors behaves

        g: grid
        """
        # upper row
        g_im1_jm1 = np.roll(np.roll(self.beh_grid[1], -1, axis=0), -1, axis=1)  # 1
        g_im1_j = np.roll(np.roll(self.beh_grid[2], -1, axis=0), 0, axis=1)  # 2
        g_im1_jp1 = np.roll(np.roll(self.beh_grid[3], -1, axis=0), +1, axis=1)  # 3
        # same row
        g_i_jm1 = np.roll(np.roll(self.beh_grid[4], 0, axis=0), -1, axis=1)  # 4
        g_i_jp1 = np.roll(np.roll(self.beh_grid[5], 0, axis=0), +1, axis=1)  # 5
        # lower row
        g_ip1_jm1 = np.roll(np.roll(self.beh_grid[6], +1, axis=0), -1, axis=1)  # 6
        g_ip1_j = np.roll(np.roll(self.beh_grid[7], +1, axis=0), 0, axis=1)  # 7
        g_ip1_jp1 = np.roll(np.roll(self.beh_grid[8], +1, axis=0), +1, axis=1)  # 8

        beh_ngb = np.array([self.beh_grid[0],
                            g_im1_jm1, g_im1_j, g_im1_jp1,
                            g_i_jm1, g_i_jp1,
                            g_ip1_jm1, g_ip1_j, g_ip1_jp1])
        return beh_ngb

    @staticmethod
    def compare_to_ngb(matrix_to_roll):
        """
        :param matrix_to_roll:
        :return:
        """
        # upper row
        g_im1_jm1 = np.roll(np.roll(matrix_to_roll, -1, axis=0), -1, axis=1)
        g_im1_j = np.roll(np.roll(matrix_to_roll, -1, axis=0), 0, axis=1)
        g_im1_jp1 = np.roll(np.roll(matrix_to_roll, -1, axis=0), +1, axis=1)
        # same row
        g_i_jm1 = np.roll(np.roll(matrix_to_roll, 0, axis=0), -1, axis=1)
        g_i_jp1 = np.roll(np.roll(matrix_to_roll, 0, axis=0), +1, axis=1)
        # lower row
        g_ip1_jm1 = np.roll(np.roll(matrix_to_roll, +1, axis=0), -1, axis=1)
        g_ip1_j = np.roll(np.roll(matrix_to_roll, +1, axis=0), 0, axis=1)
        g_ip1_jp1 = np.roll(np.roll(matrix_to_roll, +1, axis=0), +1, axis=1)


        matrix_ngb = [matrix_to_roll,
                      g_im1_jm1, g_im1_j, g_im1_jp1,
                      g_i_jm1, g_i_jp1,
                      g_ip1_jm1, g_ip1_j, g_ip1_jp1]
        return matrix_ngb

    # --------------------------------------- PAYOFF GRID --------------------------------------- #
    def calculate_one_payoff(self, my_matrix, one_ngb):
        """
        This function shows and describes how would look like the results of one game.
        So for example: shows the reward for player which both of them cooperates with
        each other.

        :param my_matrix: matrix contains behaviour of player's in that.
            (array: square matrix)
        :param one_ngb: matrix contains one of neighbours to players. So that matrix you can
            get by rolling the  `my_matrix`.
            (array: square matrix)
        :return: payoff's matrix, which presents results of one game for all players.
        """
        state_cc = self.payoff_cc * (np.logical_and(my_matrix == 'C', one_ngb == 'C')).astype(int)
        state_cd = self.payoff_cd * (np.logical_and(my_matrix == 'C', one_ngb == 'D')).astype(int)
        state_dc = self.payoff_dc * (np.logical_and(my_matrix == 'D', one_ngb == 'C')).astype(int)
        state_dd = self.payoff_dd * (np.logical_and(my_matrix == 'D', one_ngb == 'D')).astype(int)

        return state_cc + state_cd + state_dc + state_dd

    def create_payoff_grid(self):
        """
        :return:
        """

        for i in range(1, self.n_ngb):
            # we take `1` instead of `0` because we do not have play with ourselves.
            # I have to fixed my and my neighbour behaviour
            my_beh = self.beh_grid[i]
            my_ngb_beh = self.beh_ngb_now[i]
            # calculate payoffs
            one_payoff = self.calculate_one_payoff(my_beh, my_ngb_beh)
            self.payoff_grid += one_payoff

    # --------------------------------------- CHANGE STRATEGY --------------------------------------- #
    def change_strategy(self):
        """
        :return:
        """
        # which of our neighbours have the best payoff
        idnex_best_ngb = np.array([[None] * self.n_size] * self.n_size)
        # check and searched for 'strongest' neighbourhood
        payoff_grid_roll = self.compare_to_ngb(self.payoff_grid)
        for i in range(0, self.n_ngb - 1):
            idnex_best_ngb = np.where(payoff_grid_roll[i] > payoff_grid_roll[i+1], i, i+1)

        # changing strategy
        str_grid_roll = self.compare_to_ngb(self.str_grid)
        for i in range(0, self.n_ngb):
            self.str_grid = np.where(idnex_best_ngb == i, str_grid_roll[i], self.str_grid)

    # --------------------------------------- SET BEHAVIOUR: AFTER GAME --------------------------------------- #
    def set_beh_after_play(self):
        """
        :return:
        """
        for i in range(0, self.n_ngb):
            # --- set behaviour of: pavlov (version 2)
            # condition to win
            do_I_won_con1 = np.where(np.logical_and(self.beh_grid[i] == 'C', self.beh_ngb_now[i] == 'C'), 1, 0)
            do_I_won_con2 = np.where(np.logical_and(self.beh_grid[i] == 'D', self.beh_ngb_now[i] == 'C'), 1, 0)
            do_I_won = np.where(np.logical_or(do_I_won_con1 == 1, do_I_won_con2 == 1), 1, 0)
            # swap whole grid
            swap_my_beh = np.where(self.beh_grid[i].copy() == 'C', 'D', 'C')
            # change Pavlov behaviour
            self.beh_grid[i] = np.where(np.logical_and(self.str_grid == 'Pav', do_I_won == 0),
                                        swap_my_beh, self.beh_grid[i])

            # --- set behaviour of: cooperators
            self.beh_grid[i] = np.where(self.str_grid == 'Cop', 'C', self.beh_grid[i])

            # --- set behaviour of: deflectors
            self.beh_grid[i] = np.where(self.str_grid == 'Def', 'D', self.beh_grid[i])

            # --- set behaviour of: tit-for-tap
            self.beh_grid[i] = np.where(self.str_grid == 'Tit',
                                        self.beh_ngb_now[i].copy(),
                                        self.beh_grid[i])

    # ##################################### DO ONE STEP ##################################### #
    def do_one_step(self):
        # --- firstly we play few games with our neighbours
        for i in range(0, self.m_games):
            self.beh_ngb_now = self.create_beh_ngb_now()  # How look new neighbour behaviour
            self.create_payoff_grid()  # Playing games: payoffs
            self.set_beh_after_play()  # How will look like my behaviour in next game

        # --- after games we have to adapt new strategy
        self.change_strategy()  # Look up my neighbours, how my strategy should look like
        self.set_beh_after_play()  # How will look like my behaviour in next game
        self.payoff_grid = np.array([[0.0] * self.n_size] * self.n_size)  # after play, we reset payoffs

    # --------------------------------------- RETURN DATA --------------------------------------- #
    def return_strategy_grid(self):
        return self.str_grid.copy()

    def return_behaviour_grid(self):
        return self.beh_grid.copy()

    def return_payoff_grid(self):
        return self.payoff_grid.copy()

    def return_behaviour_ngb_now(self):
        return self.beh_ngb_now.copy()

    def do_color_coding(self):
        """
        :return:
        """
        # print("self.str_grid")
        # pprint(self.str_grid)

        state_cc = 1.0 * (np.where(self.str_grid == 'Cop', 1, 0)).astype(int)
        state_cd = 2.0 * (np.where(self.str_grid == 'Def', 1, 0)).astype(int)
        state_dc = 3.0 * (np.where(self.str_grid == 'Pav', 1, 0)).astype(int)
        state_dd = 4.0 * (np.where(self.str_grid == 'Tit', 1, 0)).astype(int)

        state = np.array(state_cc + state_cd + state_dc + state_dd)
        return state.copy()


if __name__ == '__main__':
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


    def create_initial_lattice(_n_size):
        """
        This function will create a lattice which create a grid, where 25% of players
        are cooperators, 25% are defectors, 25% are pavlov and 25% are tit-for-tap.

        :param _n_size: the size (dimension) of square lattice of our interest
            (array: square)
        :return: lattice with cooperators and defectors.
            (array: square)
        """
        init_lattice = np.array([[None for x in range(n_size)] for y in range(n_size)])
        flat_indexes_defectors = np.random.choice(n_size * n_size, n_size * n_size, replace=False)

        for player in range(0, int(_n_size**2)):
            index = flat_indexes_defectors[player]
            index_i, index_j = get_index_2d(index, n_size)

            if player < int(_n_size**2 / 4):
                # set cooperators
                init_lattice[index_i][index_j] = 'Cop'
            elif player < int(2 * _n_size**2 / 4):
                # set deflectors
                init_lattice[index_i][index_j] = 'Def'
            elif player < int(3 * _n_size**2 / 4):
                # set pavlov
                init_lattice[index_i][index_j] = 'Pav'
            else:
                # set 'tit-for-tap'
                init_lattice[index_i][index_j] = 'Tit'

        return init_lattice

    # ----------------------- INITIAL SETS ----------------------- #
    output_path = "./ExtraTask"
    make_directory(output_path)
    n_size = 201
    steps = 100
    val_b = 1.5
    # initial grid with same number of players with differents strategy
    init_grid = create_initial_lattice(n_size)
    pprint(init_grid)
    print()
    # initialise the class
    class_PrisonDilemma = PrisonDilemma(init_grid, val_b=val_b)

    for step in range(0, steps):

        print('I m doing step:', step)

        # do one step
        class_PrisonDilemma.do_one_step()
        # --------------------------- PLOT: LATTICE --------------------------- #
        after_color_coding = class_PrisonDilemma.do_color_coding()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(after_color_coding, interpolation='nearest', cmap='viridis')
        cax.set_clim(vmin=1, vmax=4)
        cbar = fig.colorbar(cax, ticks=[1.0, 2.0, 3.0, 4.0], orientation='vertical')
        # plt.show()
        napis = f'{step:03d}'

        plt.savefig(output_path + '/Prison-dilemma-step-' + napis + '.png', dpi=300)
        plt.close(fig)

    # --- make gif
    # make_gif(output_path)
