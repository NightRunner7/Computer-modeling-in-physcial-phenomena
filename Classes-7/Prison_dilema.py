import numpy as np
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

        `C`: describes the cooperative player (or with cooperative strategy) - always cooperative
        `D`: describes the defection player (or with defection strategy) - always defecting

    Thus, that means our lattice will have in each cell: `C` or `D`.
    """

    def __init__(self, init_lattice, val_b=1.0):
        """
        :param init_lattice: lattice or grid, which describes initial condition in our case.
            That means it is an array, which has elements: `C` or `D`, to differentiate the
            strategy of player, localised in cell.
            (array: looks like square matrix)
        :param val_b: the parameter, which will change. This parameter describes the `payoff`
            or reward on betray someone of his neighbours.
            (float)

        evo: evolution
        lat: lattice
        """
        # input parameters
        self.init_lat = init_lattice
        self.n_size = len(init_lattice)
        self.lat_results_one_game = np.array([[None for x in range(self.n_size)] for y in range(self.n_size)])
        # payoffs: describes the results or reward of one game
        self.payoff_cc = 1  # I being cooperative and my opponent being cooperative
        self.payoff_cd = 0  # I being cooperative, but my opponent defected me
        self.payoff_dc = val_b  # I being deflated and my opponent being cooperative
        self.payoff_dd = 0  # I being deflated as my opponent
        # Neighbourhood state: describes the payoffs coming from nine game
        self.NbgState = np.array([[None for x in range(self.n_size)] for y in range(self.n_size)])
        # evolution player behaviour
        self.evo_behaviour_lat = np.array([[None for x in range(self.n_size)] for y in range(self.n_size)])

    # --------------------------------------- NEIGHBOURHOOD STATE --------------------------------------- #
    def calculate_one_payoff(self, my_matrix, one_ngb):
        """
        This function shows and describes how would look like the results of one game.
        So for example: shows the reward for player which both of them cooperates with
        each other.

        :param my_matrix: matrix contains player's with fixed strategy.
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

    def neighbourhood_state(self, lattice):
        """
        This function creates a matrix, which contains information: how does nine games
        goes for each player? He has get a high reward? Or he loses?

        :param lattice: 2D list contains `C` and `D`.
            (array: square matrix)
        :return: Nothing. Just update who looks the matrix, which describes results of nine games
            for each player (each cell). Any element of that matrix is a float.
            (array: square matrix)
        """
        # upper row
        l_im1_jm1 = np.roll(np.roll(lattice, -1, axis=0), -1, axis=1)
        l_im1_j = np.roll(np.roll(lattice, -1, axis=0), 0, axis=1)
        l_im1_jp1 = np.roll(np.roll(lattice, -1, axis=0), +1, axis=1)
        # same row
        l_i_jm1 = np.roll(np.roll(lattice, 0, axis=0), -1, axis=1)
        l_i_j = np.roll(np.roll(lattice, 0, axis=0), 0, axis=1)  # HOW WE ARE!
        l_i_jp1 = np.roll(np.roll(lattice, 0, axis=0), +1, axis=1)
        # lower row
        l_ip1_jm1 = np.roll(np.roll(lattice, +1, axis=0), -1, axis=1)
        l_ip1_j = np.roll(np.roll(lattice, +1, axis=0), 0, axis=1)
        l_ip1_jp1 = np.roll(np.roll(lattice, +1, axis=0), +1, axis=1)

        # For each cell we have 9 neighbours (also ourselves): we played nine games
        payoff_ngb_1 = self.calculate_one_payoff(l_i_j, l_im1_jm1)
        payoff_ngb_2 = self.calculate_one_payoff(l_i_j, l_im1_j)
        payoff_ngb_3 = self.calculate_one_payoff(l_i_j, l_im1_jp1)
        payoff_ngb_4 = self.calculate_one_payoff(l_i_j, l_i_jm1)
        payoff_me = self.calculate_one_payoff(l_i_j, l_i_j)
        payoff_ngb_5 = self.calculate_one_payoff(l_i_j, l_i_jp1)
        payoff_ngb_6 = self.calculate_one_payoff(l_i_j, l_ip1_jm1)
        payoff_ngb_7 = self.calculate_one_payoff(l_i_j, l_ip1_j)
        payoff_ngb_8 = self.calculate_one_payoff(l_i_j, l_ip1_jp1)

        NbgState = payoff_ngb_1 + payoff_ngb_2 + payoff_ngb_3 + payoff_ngb_4 + payoff_me + \
                   payoff_ngb_5 + payoff_ngb_6 + payoff_ngb_7 + payoff_ngb_8

        self.NbgState = np.array(NbgState)

    # --------------------------------------- CHANGING STRATEGY --------------------------------------- #
    def change_strategy(self):
        """
        This function will describe how will look changing the strategy for any player. The sets of
        rules are really simple: every player after playing nine games just looks around his neighbourhood
        and looks for person, which have the biggest reward after nine games. He will just copy his
        strategy in next time step.

        :return: Nothing. Just update how will look like our lattice, which describes the strategy
            of our players.
        """
        # --- matrix with 'C' and 'D': player's strategy
        # upper row
        l_im1_jm1 = np.roll(np.roll(self.init_lat, -1, axis=0), -1, axis=1)
        l_im1_j = np.roll(np.roll(self.init_lat, -1, axis=0), 0, axis=1)
        l_im1_jp1 = np.roll(np.roll(self.init_lat, -1, axis=0), +1, axis=1)
        # same row
        l_i_jm1 = np.roll(np.roll(self.init_lat, 0, axis=0), -1, axis=1)
        l_i_j = np.roll(np.roll(self.init_lat, 0, axis=0), 0, axis=1)  # HOW WE ARE!
        l_i_jp1 = np.roll(np.roll(self.init_lat, 0, axis=0), +1, axis=1)
        # lower row
        l_ip1_jm1 = np.roll(np.roll(self.init_lat, +1, axis=0), -1, axis=1)
        l_ip1_j = np.roll(np.roll(self.init_lat, +1, axis=0), 0, axis=1)
        l_ip1_jp1 = np.roll(np.roll(self.init_lat, +1, axis=0), +1, axis=1)

        # --- payoff matrix: results of nine game
        # upper row
        payl_im1_jm1 = np.roll(np.roll(self.NbgState, -1, axis=0), -1, axis=1)
        payl_im1_j = np.roll(np.roll(self.NbgState, -1, axis=0), 0, axis=1)
        payl_im1_jp1 = np.roll(np.roll(self.NbgState, -1, axis=0), +1, axis=1)
        # same row
        payl_i_jm1 = np.roll(np.roll(self.NbgState, 0, axis=0), -1, axis=1)
        payl_i_j = np.roll(np.roll(self.NbgState, 0, axis=0), 0, axis=1)  # HOW WE ARE!
        payl_i_jp1 = np.roll(np.roll(self.NbgState, 0, axis=0), +1, axis=1)
        # lower row
        payl_ip1_jm1 = np.roll(np.roll(self.NbgState, +1, axis=0), -1, axis=1)
        payl_ip1_j = np.roll(np.roll(self.NbgState, +1, axis=0), 0, axis=1)
        payl_ip1_jp1 = np.roll(np.roll(self.NbgState, +1, axis=0), +1, axis=1)

        # neighbour index 1: comparison
        comp_1 = (payl_im1_jm1 > payl_i_j).astype(int)
        payl_i_j = np.where(comp_1 == 1, payl_im1_jm1, payl_i_j)
        l_i_j = np.where(comp_1 == 1, l_im1_jm1, l_i_j)
        # neighbour index 2: comparison
        comp_1 = (payl_im1_j > payl_i_j).astype(int)
        payl_i_j = np.where(comp_1 == 1, payl_im1_j, payl_i_j)
        l_i_j = np.where(comp_1 == 1, l_im1_j, l_i_j)
        # neighbour index 3: comparison
        comp_1 = (payl_im1_jp1 > payl_i_j).astype(int)
        payl_i_j = np.where(comp_1 == 1, payl_im1_jp1, payl_i_j)
        l_i_j = np.where(comp_1 == 1, l_im1_jp1, l_i_j)
        # neighbour index 4: comparison
        comp_1 = (payl_i_jm1 > payl_i_j).astype(int)
        payl_i_j = np.where(comp_1 == 1, payl_i_jm1, payl_i_j)
        l_i_j = np.where(comp_1 == 1, l_i_jm1, l_i_j)
        # neighbour index 5: comparison
        comp_1 = (payl_i_jp1 > payl_i_j).astype(int)
        payl_i_j = np.where(comp_1 == 1, payl_i_jp1, payl_i_j)
        l_i_j = np.where(comp_1 == 1, l_i_jp1, l_i_j)
        # neighbour index 6: comparison
        comp_1 = (payl_ip1_jm1 > payl_i_j).astype(int)
        payl_i_j = np.where(comp_1 == 1, payl_ip1_jm1, payl_i_j)
        l_i_j = np.where(comp_1 == 1, l_ip1_jm1, l_i_j)
        # neighbour index 7: comparison
        comp_1 = (payl_ip1_j > payl_i_j).astype(int)
        payl_i_j = np.where(comp_1 == 1, payl_ip1_j, payl_i_j)
        l_i_j = np.where(comp_1 == 1, l_ip1_j, l_i_j)
        # neighbour index 8: comparison
        comp_1 = (payl_ip1_jp1 > payl_i_j).astype(int)
        payl_i_j = np.where(comp_1 == 1, payl_ip1_jp1, payl_i_j)
        l_i_j = np.where(comp_1 == 1, l_ip1_jp1, l_i_j)

        self.init_lat = l_i_j
        self.NbgState = payl_i_j

    # --------------------------------------- DO ONE STEP IN EVOLUTION --------------------------------------- #
    def one_step(self):
        """
        This function will govern of playing nine games for each player and will be responsible
        for update their strategy after those games.

        :return: Nothing. Just update all necessary data and matrices.
        """
        # clear: evolution player behaviour
        self.evo_behaviour_lat = np.array([[None for x in range(self.n_size)] for y in range(self.n_size)])
        # how look strategy before playing games
        self.lat_results_one_game = self.init_lat.copy()
        # create matrix, describes who goes the nine games for each player
        self.neighbourhood_state(self.init_lat)
        # after playing nine games, each player will want to check results of his neighbours and maybe
        # change his strategy.
        self.change_strategy()
        # this matrix will present, how does strategy changes after nine games
        self.evo_behaviour_lat = np.array(
            [[self.init_lat[i][j] + self.lat_results_one_game[i][j] for i in range(self.n_size)]
             for j in range(self.n_size)])

    # --------------------------------------- INPUT DATA --------------------------------------- #
    def put_lattice(self, lattice):
        """
        :param lattice: 2D list contains `C` and `D`.
            (array: square matrix)
        :return: Nothing. Just update the data stored in class.
        """
        self.init_lat = lattice

    def update_b_parameter(self, val_b):
        """
        :param val_b: parameter, which describes the `DC` payoff.
            (float)
        :return: Nothing. Just update the data stored in class.
        """
        self.payoff_dc = val_b

    # --------------------------------------- RETURN DATA --------------------------------------- #
    def return_lat(self):
        return self.init_lat.copy()

    def return_previous_init_lat(self):
        return self.lat_results_one_game.copy()

    def return_ngb_state(self):
        return self.NbgState.copy()

    def return_color_coding(self):
        return self.evo_behaviour_lat.copy()

    def do_color_coding(self):
        """
        We want to see how looks changes in selecting strategy during our evolution.
        But the matrix `evo_behaviour_lat`, has encoded that information as string.
        For example: player with `DC` denotes:

            before the nine game, our player has stayed with strategy: `always cooperative`,
            but after nine games, he discovers that player in his neighbourhood, which get
            the best payoff or reward, sticks to strategy: `always deflecting`. So, he
            decides to change his strategy into: `always deflecting`.

        To better visualised that changing I set four different number for any possible results
        of changing strategy.

        :return:
        """
        state_cc = 1.0 * (np.where(self.evo_behaviour_lat == 'CC', 1, 0)).astype(int)
        state_cd = 2.0 * (np.where(self.evo_behaviour_lat == 'CD', 1, 0)).astype(int)
        state_dc = 3.0 * (np.where(self.evo_behaviour_lat == 'DC', 1, 0)).astype(int)
        state_dd = 4.0 * (np.where(self.evo_behaviour_lat == 'DD', 1, 0)).astype(int)

        state = state_cc + state_cd + state_dc + state_dd
        return state.copy()

    def return_ratio_deflectors(self):
        """:return: ratio of deflectors in whole lattice compare to all players in the grid."""
        only_deflectors = 1.0 * (np.where(self.init_lat == 'D', 1, 0)).astype(int)
        ratio_deflectors = np.sum(only_deflectors) / self.n_size**2
        return ratio_deflectors

if __name__ == '__main__':
    # ----------------------- DO STEPS ----------------------- #
    output_path = "./Task1_part1"
    make_directory(output_path)
    n_size = 201
    init_lattice = [['C' for x in range(n_size)] for y in range(n_size)]
    init_lattice[100][100] = 'D'
    val_b = 1.9

    class_PrisonDilemma = PrisonDilemma(init_lattice, val_b=val_b)

    steps = 100

    for i in range(0, steps):
        # do one step
        class_PrisonDilemma.one_step()
        # --------------------------- PLOT: LATTICE --------------------------- #
        after_color_coding = class_PrisonDilemma.do_color_coding()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(after_color_coding, interpolation='nearest', cmap='viridis')
        cax.set_clim(vmin=1, vmax=4)
        cbar = fig.colorbar(cax, ticks=[1.0, 2.0, 3.0, 4.0], orientation='vertical')
        # plt.show()
        napis = f'{i:03d}'

        plt.savefig(output_path + '/Prison-dilemma-step-' + napis + '.png', dpi=300)
        plt.close(fig)

    # --- make gif
    make_gif(output_path)
