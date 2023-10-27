import numpy as np
import sys
import random
import pprint
import copy

# ########################## CLASS DEFINITION ########################## #

class FormNewPopulation(object):
    def __init__(self, chromosomes_map):
        """
        :param chromosomes_map:
        """
        # initial chromosome
        self.chromes_map = chromosomes_map
        self.Nchromes = len(self.chromes_map)  # number of chromosomes
        self.length_chrom = len(self.chromes_map[0]["binaryStr"])  # length of zeros and one - length of our chromosome
        # number of population of chromosomes, we will try to have equilibrium between deaths and reproduction
        # or cloning, so the population value of chromosomes should be preserved.
        self.population_Nchromes = len(chromosomes_map)

        # after population step we will kill some chromosomes, but some of them will survive
        # we want to differentiate those. So we have somewhere stored them.
        self.parent_chromes_map = {}
        self.parent_Nchromes = None

    # ----------------------------- FUN WITH: `fitValue` ----------------------------- #
    def create_probable_list(self):
        """
        We will use only parents in this creation!

        probable value = fitValue / (sum of all fitValue)

        :return: [index_list, probable_list], first list contains the index of chromosome,
            the second list contains the probable vale
        """
        index_list = list(self.parent_chromes_map.keys())
        probable_list = []
        # get all `fitValue`
        fitValue_list = [self.parent_chromes_map[id]["fitValue"] for id, info in self.parent_chromes_map.items()]
        fitValue_sum = sum(fitValue_list)
        for i in range(0, len(fitValue_list)):
            probable = float(fitValue_list[i] / fitValue_sum)
            # append
            probable_list.append(probable)

        return [index_list.copy(), probable_list.copy()]

    # ----------------------------- KILLING CHROMOSOMES ----------------------------- #
    def kill_chromes(self, n_murder):
        """
        Kill those chromosomes, which have the worse value of fitting function.
        In the other words just kill those, which will not lead us to our goal.

        :param n_murder: number of murder's, which we have to commit.
            (int)
        :return: Nothing. Just decide to reduce the number of populations
            of our chromosomes.
        --------
        One important remark: after killing we have sorted `chromosomes map`,
        chromosomes with index `0`, have the highest value of `fitValue`.
        """
        # --- firstly we have to sort map by `fitValue`
        # important remark: sort_chromes_list is a `list` NOT `dictionary`
        sort_chromes_list = sorted(self.chromes_map.items(), key=lambda item: item[1].get("fitValue"), reverse=True)

        # --- secondly we have to commit murders
        for kill in range(0, n_murder):
            sort_chromes_list.pop()

        n_survive = len(sort_chromes_list)
        # --- thirdly we create a new map with new indexes
        new_chromes_map = {}
        for index in range(0, n_survive):
            # chromosome attributes
            chromes_atr = sort_chromes_list[index][1]
            new_chromes_map[index] = chromes_atr

        # --- update the number of chromosomes
        self.chromes_map = new_chromes_map
        self.Nchromes = len(self.chromes_map)
        # --- fixed parent chromosomes at that population: we have to do deepcopy
        self.parent_chromes_map = copy.deepcopy(new_chromes_map)
        self.parent_Nchromes = len(self.parent_chromes_map)

    def kill_by_force(self, further_killing=False, stay_alive=2):
        """
        This function is only for TEST's, is intended to kill only those
        of chromosomes, which have: `NONE` in `fitValue`

        :param further_killing: you commit NONE's (is good joke) so maybe you
            want kill someone else?
            (bool)
        :param stay_alive: how many chromosomes may survive?
            (int)
        :return: Nothing. Just decide to reduce the number of populations
            of our chromosomes.
        """
        index_list = list(self.chromes_map.keys())
        new_chromosome_map = {}
        new_id = 0
        for i in range(0, self.Nchromes):
            index = index_list[i]
            fitValue = self.chromes_map[index]["fitValue"]
            if fitValue is not None:
                new_chromosome_map[new_id] = self.chromes_map[index]
                new_id += 1
                if further_killing is True:
                    if new_id >= stay_alive:
                        break

        self.chromes_map = new_chromosome_map
        self.Nchromes = len(self.chromes_map)
        # --- fixed parent chromosomes at that population: we have to do deepcopy
        self.parent_chromes_map = copy.deepcopy(new_chromosome_map)
        self.parent_Nchromes = len(self.parent_chromes_map)

    # ----------------------------- MUTATION ----------------------------- #
    def do_mutation(self, chromes_str, mut_bits):
        """
        Do mutation for our chromosomes.

        :param chromes_str: list contains all the chromosomes string, which will go mutate
            in fixed step of population. Remark: it can be one string of zeros and ones - list,
            but it also can be a number of chromosomes, which will mutate - list of lists
            (list of lists / array of arrays) or (list / array)
        :param mut_bits: mutation bites, how many random bits of information you
            want to change in getting list of zeros and ones.
            (int)
        :return: list or list of list, which have chromosomes after step of mutations.

        Important remark: same lengths of chromosomes.
        mut: mutation
        muts: mutations
        """
        if isinstance(chromes_str, list) or isinstance(chromes_str, np.ndarray):
            # number of chromosomes, which mutated
            Nchromes = len(chromes_str)
            for chrom in range(0, Nchromes):
                # for each chromosome
                change_bits = np.random.randint(0, self.length_chrom, mut_bits)  # where mutation appears
                for mut in range(0, mut_bits):
                    mut_index = change_bits[mut]
                    # do mutation: 1->0, 0->1
                    chromes_str[chrom][mut_index] = (chromes_str[chrom][mut_index] + 1) % 2
        else:
            change_bits = np.random.randint(0, self.length_chrom, mut_bits)  # where mutation appears
            for mut in range(0, mut_bits):
                mut_index = change_bits[mut]
                # do mutation: 1->0, 0->1
                chromes_str[mut_index] = (mut_index[mut_index] + 1) % 2

        return chromes_str

    # ----------------------------- CLONING ----------------------------- #
    def clone_and_mutation_simple(self, n_parents, n_clones, mut_bits=5):
        """
        The simples way of cloning to enlarge numerous of chromosomes. Simple
        means that we consider all parents, which survive into cloning mechanism.

        :param n_parents: how many chromosomes from `paren_chromes_map` we want
            to consider during cloning.
            (int)
        :param n_clones: how many chromosomes we want to clone.
            (int)
        :param mut_bits: mutation bites, how many random bits of information you
            want to change in getting list of zeros and ones.
            (int)
        :return: Nothing. Just create new chromosomes to fulfill
            the population limit.
        """
        # --- fixed parents
        if n_parents > self.parent_Nchromes:
            sys.exit(f"You set the number of parents: {n_parents}, "
                     f"but class has been stored: {self.parent_Nchromes} parents")
        # --- Do cloning
        cloneChrom_str_list = []

        for clone in range(0, n_clones):
            # parent chromosomes we use to cloning
            index_parent = clone % n_parents
            cloneChrom_str = self.parent_chromes_map[index_parent]["binaryStr"].copy()  # .copy() IMPORTANT!
            # append
            cloneChrom_str_list.append(cloneChrom_str)

        # --- Do mutation for every clone
        cloneChrom_str_list = self.do_mutation(cloneChrom_str_list, mut_bits)

        # --- Update chromosomes map
        first_clone_id = self.Nchromes
        for clone in range(0, n_clones):
            # clone chromosomes string
            chrom_str = cloneChrom_str_list[clone]
            chrom_atr = {"binaryStr": chrom_str, "fitValue": None}
            # update chromosome map
            self.chromes_map[first_clone_id + clone] = chrom_atr

        self.Nchromes = len(self.chromes_map)

    # ----------------------------- REPRODUCTION ----------------------------- #
    def do_one_child(self, chrom_str_p1, chrom_str_p2):
        """
        In usually reproduction we need two parents chromosomes
        to create a one child.

        :param chrom_str_p1: chromosome string of first parent.
            (array or list)
        :param chrom_str_p2: chromosome string if second parent.
            (array or list)
        :return: children string chromosome.

        Important remark: parents have the same length chromosome.
        """
        child_chrom_str = np.array([], dtype=int)
        # create list contains parents chromosomes
        parents_chrom = [chrom_str_p1, chrom_str_p2]
        # list contains zeros and ones: describes from which parent
        # we have to take a bit of information.
        which_parent_list = np.random.randint(0, 2, self.length_chrom)
        for i in range(0, self.length_chrom):
            which_parent = which_parent_list[i]
            # what will be child bit
            bit = parents_chrom[which_parent][i]
            # append
            child_chrom_str = np.append(child_chrom_str, bit)

        return child_chrom_str

    def reproduction_fitvalue(self, child_chrom):
        """
        How probable will be reproduction depends on value of `fitValue`.
        We have to take care that all potential parents have positive
        value of `fitValue`.

        :param child_chrom: how many children we want to have.
            (int)
        :return: Nothing. Just create new chromosomes to fulfill
            and update the stored dictionary of chromosomes.
        """
        # --- get data
        # id / indexes of parents and the probability of finding partner
        index_list, probable_list = self.create_probable_list()

        for child in range(0, child_chrom):
            # --- pick up parents
            # find first parent
            parent1_id = random.choices(index_list, cum_weights=probable_list, k=1)[0]
            # Remove the selected number from the list to ensure the second number is different
            parent1_probable = probable_list[parent1_id]
            index_list.pop(parent1_id)
            probable_list.pop(parent1_id)
            # find second parent
            parent2_id = random.choices(index_list, cum_weights=probable_list, k=1)[0]

            # after choice second parent, we have added back first parent into right place
            index_list.insert(parent1_id, parent1_id)
            probable_list.insert(parent1_id, parent1_probable)

            # --- create a one child
            chrom_str_p1 = self.chromes_map[parent1_id]["binaryStr"]
            chrom_str_p2 = self.chromes_map[parent2_id]["binaryStr"]
            child_chrom_str = self.do_one_child(chrom_str_p1, chrom_str_p2)

            # --- update the stored chromosomes
            self.chromes_map[self.Nchromes] = {"binaryStr": child_chrom_str, "fitValue": None}
            self.Nchromes += 1

    # ------------------------------- RETURN/UPDATE DATA ------------------------------- #
    def return_chromes_map(self):
        """REALLY IMPORTANT! If we want to do not return as reference, we have to return as deepcopy."""
        return copy.deepcopy(self.chromes_map)

    def return_fitvalue(self):
        fitValues_list = np.array([])
        for chrom in range(0, self.Nchromes):
            fitValue = self.chromes_map[chrom]["fitValue"]
            fitValues_list = np.append(fitValues_list, fitValue)
        return fitValues_list

    def return_fitvalue_parent(self):
        fitValues_list = np.array([])
        for chrom in range(0, self.parent_Nchromes):
            fitValue = self.parent_chromes_map[chrom]["fitValue"]
            fitValues_list = np.append(fitValues_list, fitValue)
        return fitValues_list

    def update_chromes_map(self, new_chromes_map):
        """
        Update chromosomes. In update, I mean use another function / class
        to calculate the value of: `fitValue`.

        :param new_chromes_map: map of chromosomes, after offspring or reproduction,
            or so on. This map should store all necessary information, which
            we use in our class. Moore details in `self.initial_chromosome`
            (dictionary)
        :return: Nothing. Just update the chromosomes stored in this class.
        """
        self.chromes_map = new_chromes_map
        self.Nchromes = len(new_chromes_map)


if __name__ == '__main__':
    # --- testing: create example list of chromosomes
    nchrom = 40
    chrom_len = 10  # length of one chromosome
    chromes_map = {}  # dictionary, which will contains all necessary information of chromosomes

    for chrom in range(nchrom):
        # random chromosome `string`
        chrom_str = np.random.randint(2, size=chrom_len)
        # fitValue
        fitValue = np.random.randint(-20, 50)
        # chromosome attributes
        chrom_atr = {"binaryStr": chrom_str, "fitValue": fitValue}
        chromes_map[chrom] = chrom_atr

    pprint.pprint(chromes_map)
    sorted_chromes_map = sorted(chromes_map.items(), key=lambda item: item[1].get("fitValue"), reverse=True)
    print()
    print("after sorting:")
    pprint.pprint(sorted_chromes_map)

    # --- create a class like object
    NewChromosomes = FormNewPopulation(chromes_map)

    # --- checkin the working of `kill_chromes`
    murders = 30
    NewChromosomes.kill_chromes(murders)

    chromes_list = NewChromosomes.return_chromes_map()
    print()
    print("After killing")
    pprint.pprint(chromes_list)

    # --- checkin the working of `clone_and_mutation_simple`
    NewChromosomes.clone_and_mutation_simple(n_parents=5, n_clones=30, mut_bits=5)

    chromes_list = NewChromosomes.return_chromes_map()
    print()
    print("After cloning")
    pprint.pprint(chromes_list)

    # --- checkin the working of `kill_by_force`: we have to do that because born chromosomes
    # do not have vale of `fitValue`: only two parents will be alive
    NewChromosomes.kill_by_force(further_killing=True, stay_alive=2)

    chromes_list = NewChromosomes.return_chromes_map()
    print()
    print("After kill_by_force")
    pprint.pprint(chromes_list)

    # --- checkin the working of reproduction_fitvalue
    childs = 10
    NewChromosomes.reproduction_fitvalue(childs)

    chromes_list = NewChromosomes.return_chromes_map()
    print()
    print("After reproduction")
    pprint.pprint(chromes_list)

    # --- checkin the working of `kill_by_force`: we have to do that because born chromosomes
    # do not have vale of `fitValue`: We want only to have one: easier to check the `clone_and_mutation_simple`.
    NewChromosomes.kill_by_force(further_killing=True, stay_alive=1)

    chromes_list = NewChromosomes.return_chromes_map()
    print()
    print("After kill_by_force")
    pprint.pprint(chromes_list)

    # --- checkin the working of `clone_and_mutation_simple`
    NewChromosomes.clone_and_mutation_simple(n_parents=1, n_clones=20, mut_bits=2)

    chromes_list = NewChromosomes.return_chromes_map()
    print()
    print("After cloning")
    pprint.pprint(chromes_list)
