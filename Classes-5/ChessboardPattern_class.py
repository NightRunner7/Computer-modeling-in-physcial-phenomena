import numpy as np
import copy

# ########################## CLASS DEFINITION ########################## #

class ChessboardPattern(object):
    """
    One important remark: This whole class does not fix the sharing
    chromosomes, that should be implemented in different class or
    code / file. This class only governs doing evolution, but does not
    say anything about process of sharing.
    """
    def __init__(self, n_size=50, init_lattices=5, nchrom=50):
        """
        :param n_size: the size of lattice, which you want to create. We interested in
            square lattice: our goal is the chessboard.
        :param init_lattices: how many 2D lattice you want to create.
        :param nchrom: Number of chromosomes, which we will want to create.
        """
        # initial lattice
        initial_lattices = np.array(self.random_lattice(n_size, init_lattices))
        self.init_lats = initial_lattices
        self.Nlats = init_lattices  # Number of initial lattices
        self.Nrow = len(self.init_lats[0])  # Number of rows: i
        self.Ncol = len(self.init_lats[0][0])  # Number of columns: j
        # initial chromosome
        self.chromes_map = None
        self.Nchromes = 0
        self.initial_chromosome(nchrom)

        print('len(chrom_list)', self.Nchromes)
        print('number of initial lattices:', len(initial_lattices))
        print('Nrows:', self.Nrow)
        print('Ncol:', self.Ncol)

    # -------------------------------------- INITIAL CHROMOSOMES fUNCTIONS -------------------------------------- #
    def initial_chromosome(self, nchrom=50):
        """
        Now we will crate and stored the dictionary (weaker version of c++ map in python).
        This dictionary will store all necessary information about each chromosome as we need.
        All necessary information about chromosome:

            "binaryStr": the array of zeros and identity, which have fixed length: `512`.
                (array)
            "fitValue": the value, which describes how good this chromosome is in creating
                the chessboard after `100` hundred steps (or sth like that).
                (float)

        :param nchrom: Number of chromosomes, which we will want to create.
        -----------
        chromes: denotation to the `chromosomes`.
        """
        chrom_len = 512  # length of one chromosome
        chromes_map = {}  # dictionary, which will contains all necessary information of chromosomes

        for chrom in range(nchrom):
            # random chromosome `string`
            chrom_str = np.random.randint(2, size=chrom_len)
            # chromosome attributes
            chromes_atr = {"binaryStr": chrom_str, "fitValue": None}
            chromes_map[chrom] = chromes_atr

        self.chromes_map = chromes_map
        self.Nchromes = nchrom

    # -------------------------------------- STATIC fUNCTIONS -------------------------------------- #
    @staticmethod
    def random_lattice(n_size=50, init_lattices=5):
        """
        :param n_size: the size of lattice, which you want to create. We interested in
            square lattice: our goal is the chessboard.
            (int)
        :param init_lattices: how many 2D lattice you want to create.
            (int)
        :return: the list of 2D lattices.

        lat: lattice
        """
        init_lat_all = []
        for time in range(0, init_lattices):
            init_lat = np.random.randint(2, size=(n_size, n_size))
            init_lat_all.append(init_lat)

        return init_lat_all

    @staticmethod
    def neighbourhood_state(lattice):
        """
        :param lattice: 2D list contains `1` and `0`. Is the lattice, which we're going to convert
            into chessboard.
        :return: the 2D list, which in each cell have number from `0` to `511`: the number
            depend on the neighbours of fixed cell. The cell number is `N`. Depend on that
            number we will describe the evolution.
        """
        # upper row
        l_im1_jm1 = 2**0 * np.roll(np.roll(lattice, -1, axis=0), -1, axis=1)
        l_im1_j = 2**1 * np.roll(np.roll(lattice, -1, axis=0), 0, axis=1)
        l_im1_jp1 = 2**2 * np.roll(np.roll(lattice, -1, axis=0), +1, axis=1)
        # same row
        l_i_jm1 = 2**3 * np.roll(np.roll(lattice, 0, axis=0), -1, axis=1)
        l_i_j = 2**4 * np.roll(np.roll(lattice, 0, axis=0), 0, axis=1)
        l_i_jp1 = 2**5 * np.roll(np.roll(lattice, 0, axis=0), +1, axis=1)
        # lower row
        l_ip1_jm1 = 2**6 * np.roll(np.roll(lattice, +1, axis=0), -1, axis=1)
        l_ip1_j = 2**7 * np.roll(np.roll(lattice, +1, axis=0), 0, axis=1)
        l_ip1_jp1 = 2**8 * np.roll(np.roll(lattice, +1, axis=0), +1, axis=1)

        # Neighbourhood state
        NbgState = l_im1_jm1 + l_im1_j + l_im1_jp1 + l_i_jm1 + l_i_j + l_i_jp1 + l_ip1_jm1 + l_ip1_j + l_ip1_jp1
        return NbgState

    # -------------------------------------- FITTING FUNCTION -------------------------------------- #
    @staticmethod
    def cal_fitting_one(d2_lat):
        """
        Now I say some words about our algorithm of calculating the fitting function.
        This algorithm will return number, the higher the value of this number, the greater
        the similarity to chessboard.

        :param d2_lat: lattice, which we want to compare to chessboard. fixed!
            (array of `1` and `0`)
        :return: number which describes the similarity to the chessboard.
            (float)
        ---------
        This is how look like typical 3x3 matrix: around our selected point: (i, j)

        [[(i-1, j-1), (i-1, j), (i-1, j+1)],
         [( i , j-1), ( i , j), ( i , j+1)],
         [(i+1, j-1), (i+1, j), (i+1, j+1)]]

        (1) Firstly we want tu punish each point in the lattice, which:
            state of (i, j) == state of (i+1,  j ) or
            state of (i, j) == state of ( i , j+1)
        If at least one of these conditions is true, we will punish those cells by adding: `-3 points`
        to the full pot.

        In the code we will use this description:
            state of (i+1,  j ) == go_one_down
            state of ( i , j+1) == go_one_right
        ---------
        (2) Secondly, for each cell, where any of those conditions was not fulfilled, we will consider
            state of (i, j) == state of (i+1,  j+1) or
            state of (i, j) == state of (i+1,  j-1)
        Whenever one of these conditions is True, we will award every cell by adding: `+8 points`
        to the full pot. However, each time the condition was not fulfilled, we're adding: `-5 points`
        to the full pot.

        In the code we will use this description:
            state of (i+1,  j+1) == go_one_down_right
            state of (i+1,  j-1) == go_one_down_left
        """
        n_size = len(d2_lat)  # size of lattice of our interest
        # --------------------- REALISATION OF (1) --------------------- #
        # cells, which we consider
        go_one_down = np.roll(np.roll(d2_lat, -1, axis=0), 0, axis=1)
        go_one_right = np.roll(np.roll(d2_lat, 0, axis=0), -1, axis=1)
        # compare to our fixed point
        sel_1_down = (d2_lat == go_one_down).astype(int)
        sel_1_right = (d2_lat == go_one_right).astype(int)
        # we want to consider one of those condition
        sel_1_both = np.array(sel_1_down + sel_1_right, dtype=int)
        sel_1 = np.where(sel_1_both >= 1, 1, 0)
        # now we will punish cells
        chess_lat_1 = np.array(sel_1 * (-3))

        # --------------------- REALISATION OF (2) --------------------- #
        # we want consider only those cells, which do not take into account at (1)
        after_1 = (np.zeros((n_size, n_size)) == sel_1).astype(int)
        # cells, which we consider
        go_one_down_left = np.roll(np.roll(d2_lat, -1, axis=0), +1, axis=1)
        go_one_down_right = np.roll(np.roll(d2_lat, -1, axis=0), -1, axis=1)
        # compare to our fixed point
        sel_2_downLeft = (d2_lat == go_one_down_left).astype(int)
        sel_2_downRight = (d2_lat == go_one_down_right).astype(int)
        # reward by `+8` points
        sel_2p_downLeft = (np.logical_and(after_1 == sel_2_downLeft, after_1 == 1)).astype(int)
        sel_2p_downRight = (np.logical_and(after_1 == sel_2_downRight, after_1 == 1)).astype(int)
        # punish by `-5 points`
        sel_2m_downLeft = (np.logical_and(after_1 != sel_2_downLeft, after_1 == 1)).astype(int)
        sel_2m_downRight = (np.logical_and(after_1 != sel_2_downRight, after_1 == 1)).astype(int)
        # calculate points
        chess_lat_2 = np.array(sel_2p_downLeft * 8 + sel_2p_downRight * 8 +
                               sel_2m_downLeft * (-5) + sel_2m_downRight * (-5), dtype=int)

        chess_lat = chess_lat_1 + chess_lat_2
        return sum(map(sum, chess_lat))  # summing all elements in 2D list

    def cal_fitting(self, steps_to_average=6):
        """
        Calculate fitting function for all stored chromosomes! After one generation
        we have to check how our chromosomes are doing well. The higher the: `fitValue`
        is the better are doing (better way of creating chessboard).

        :param steps_to_average: over how many last steps do you want to average.
            (int)
        :return: nothing. Update the value of `fitValue`. Stored in `self.chromes_map`
        ----------
        For more description of chromosome map go to the: `self.initial_chromosome`
        For more description of calculation fitting function go to the: `self.cal_fitting_one`
        """
        for chrom in range(0, self.Nchromes):
            # chromosome string, which we consider
            chrom_str = self.chromes_map[chrom]["binaryStr"]
            fitValue_list = []

            for lat_index in range(0, self.Nlats):
                # we consider few initial lattices

                for steps in range(0, steps_to_average):
                    # do one step
                    self.one_step_fixed(lat_index, chrom_str)
                    # calculate `fitValue`
                    lattice = self.init_lats[lat_index]
                    fitValue = self.cal_fitting_one(lattice)
                    # append to the list
                    fitValue_list.append(fitValue)

            Nelems = len(fitValue_list)
            avg_fitValue = np.sum(fitValue_list) / Nelems
            self.chromes_map[chrom]["fitValue"] = avg_fitValue

    # -------------------------------------- EVOLUTION: DO POPULATION -------------------------------------- #
    def one_step_fixed(self, index_lattice, chrom_str):
        """
        Do one step in evolution with fixed chromosome and fixed initial
        lattice.

        :param index_lattice: index of initial lattice, which refers to
            stored initial lattice in the class (usually we will use `5`)
            (int)
        :param chrom_str: chromosome string, which we consider. We will
            grow it: our set of chromosomes will change in time.
        :return: nothing. Just do one evolution step and update lattice
            after that one step.
        """
        # selected lattice
        lattice = self.init_lats[index_lattice]
        # Neighbourhood state
        NgbState = self.neighbourhood_state(lattice)
        # use chromosome in evolution
        new_lat = chrom_str[NgbState]
        # update lattice
        self.init_lats[index_lattice] = new_lat

    def one_population(self, time_steps=100, steps_to_average=6):
        """
        :param time_steps: time steps, which you want to consider (approx `100`).
            (int)
        :param steps_to_average: over how many last steps do you want to average.
            During computation `fitValue`.
            (int)
        :return: nothing just do one population epoch. The main point of our evolution
            or grow cellular automata.
        """
        # --- firstly we do around `100` time steps
        for chrom in range(0, self.Nchromes):
            # for every fixed chromosome we do evolution separately
            # chromosome string, which we consider
            chrom_str = self.chromes_map[chrom]["binaryStr"]

            for lat_index in range(0, self.Nlats):
                # we consider few initial lattices

                for step in range(0, time_steps):
                    self.one_step_fixed(lat_index, chrom_str)

        # --- secondly we compute and update `fitValue`
        self.cal_fitting(steps_to_average)

    # ------------------------------- RETURN/UPDATE DATA -------------------------------#
    def return_lattice(self, index_lattice):
        """REALLY IMPORTANT! If we want to do not return as reference, we have to return as deepcopy."""
        return copy.deepcopy(self.init_lats[index_lattice])

    def return_chromosomes_map(self):
        """REALLY IMPORTANT! If we want to do not return as reference, we have to return as deepcopy."""
        return copy.deepcopy(self.chromes_map)

    def return_neighbourhood_state(self, index_lattice):
        NgbState = self.neighbourhood_state(self.init_lats[index_lattice])
        return NgbState

    def return_fitvalue(self):
        fitValues_list = np.array([])
        for chrom in range(0, self.Nchromes):
            fitValue = self.chromes_map[chrom]["fitValue"]
            fitValues_list = np.append(fitValues_list, fitValue)
        return fitValues_list.copy()

    def update_chromosomes_map(self, chromes_after):
        """
        Update chromosomes. In update, I mean after cloning, offspring,
        reproduction and mutations.

        One important remark: This whole class does not fix the sharing
        chromosomes, that should be implemented in different class or
        code / file. This class only governs doing evolution, but does not
        say anything about process of sharing.
        -------------
        :param chromes_after: map of chromosomes, after offspring or reproduction,
            or so on. This map should store all necessary information, which
            we use in our class. Moore details in `self.initial_chromosome`
            (dictionary)
        :return: Nothing. Just update the chromosomes stored in this class.
        """
        self.chromes_map = chromes_after
        self.Nchromes = len(chromes_after)

    def update_initial_lattice(self, n_size=50, init_lattices=5):
        """
        After do one population / epoch, we usually change the initial
        lattice, which we use to train our chromosomes.

        :param n_size: the size of lattice, which you want to create. We interested in
            square lattice: our goal is the chessboard.
            (int)
        :param init_lattices: how many 2D lattice you want to create.
            (int)
        :return: Nothing. Just update the initial lattices, which we used
            in training our chromosomes.
        """
        lattice_list = self.random_lattice(n_size, init_lattices)
        self.init_lats = lattice_list
        self.Nlats = init_lattices
        self.Nrow = len(lattice_list[0])  # Number of rows: i
        self.Ncol = len(lattice_list[0][0])  # Number of columns: j

    def update_initial_lattice_vol2(self, lattice_list):
        """
        After do one population / epoch, we usually change the initial
        lattice, which we use to train our chromosomes.

        :param lattice_list: list of 2D arrays, which have zeroes and ones,
            which describes our lattice.
            (list of 2D array)
        :return: Nothing. Just update the initial lattices, which we used
            in training our chromosomes.
        """
        self.init_lats = lattice_list
        self.Nlats = len(lattice_list)
        self.Nrow = len(lattice_list[0])  # Number of rows: i
        self.Ncol = len(lattice_list[0][0])  # Number of columns: j
