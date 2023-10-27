import numpy as np
import pprint
import csv  # save file
# import sys
import matplotlib.pyplot as plt
import os
currentdir = os.getcwd()
from ChessboardPattern_class import ChessboardPattern
from offspring_chromosomes import FormNewPopulation
# ------------------------ MAKE GIF ------------------------ #
import glob
from PIL import Image

def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.PNG")]
    frame_one = frames[0]
    frame_one.save(f"{frame_folder}/Chess-Evolution.gif", format="GIF", append_images=frames,
                   save_all=True, duration=30, loop=0)

# ------------------------------- HELPFUL FUNCTION -------------------------------#
def make_directory(_output_plots_path):
    """
    :param _output_plots_path:
    :return:
    """
    try:
        # Create target Directory
        os.mkdir(_output_plots_path)
        print("Directory ", _output_plots_path,  " Created ")
    except FileExistsError:
        print("Directory ", _output_plots_path,  " already exists")

def cal_fit_part_of_max(_fitValue_list, _n_size):
    """
    :param _fitValue_list:
    :param _n_size:
    :return:
    """
    fitVale_max = _n_size ** 2 * 16
    fitVale_max_list = max(_fitValue_list)

    return fitVale_max_list / fitVale_max

# ------------------------------- MAIN EVOLUTION -------------------------------#
# --- initiate class governs evolution / population steps
Nchrom = 50  # number of chromosomes
n_size = 100  # size of lattice
output_plots_path = currentdir + '/best_Test3'
# make directory
make_directory(output_plots_path)

ChessEvolution = ChessboardPattern(n_size=n_size, init_lattices=5, nchrom=Nchrom)
# take initial chromosomes
init_chromes_map = ChessEvolution.return_chromosomes_map()

# --- initiate class governs change chromosomes population
NewPopulation = FormNewPopulation(init_chromes_map)

# --- constants of our evolution
NumsOfPops = 120  # numbers of population, which we consider
stage1_pop = 20  # different sets of cloning
stage2_pop = 50  # when we start do reproduction instead cloning

# --- stored data
fitValue_part_list = np.array([])
pop_list = np.arange(0, NumsOfPops, 1, dtype=int)
last_chromes_map = {}

for pop in range(0, NumsOfPops):
    print("population:", pop)
    # --------------------------- class `ChessboardPattern` --------------------------- #
    # firstly we randomise initial lattices
    ChessEvolution.update_initial_lattice(n_size=n_size, init_lattices=5)
    # do a population step
    ChessEvolution.one_population(time_steps=100, steps_to_average=6)
    # after that return chromosomes map
    chromes_map = ChessEvolution.return_chromosomes_map()

    # --- print
    list_print = ChessEvolution.return_fitvalue()
    print("Before killing: fitValues")
    pprint.pprint(list_print)

    # --------------------------- get `fitValue` --------------------------- #
    # take data
    fitValue_list = ChessEvolution.return_fitvalue()
    # calculate
    fitValue_part = cal_fit_part_of_max(fitValue_list, n_size)
    # append
    fitValue_part_list = np.append(fitValue_part_list, fitValue_part)

    # --------------------------- class `FormNewPopulation` --------------------------- #
    NewPopulation.update_chromes_map(chromes_map)  # update chromosomes
    if pop < stage1_pop:
        # ##### STAGE ONE IN EVOLUTION ##### #
        # Description: high number of deaths, only cloning, medium number of mutations
        # --- settings and killing
        kills = 48
        Nclones = kills  # how many clones we have to make
        NewPopulation.kill_chromes(n_murder=kills)  # killing worse
        Nparents = NewPopulation.parent_Nchromes  # number of parents
        mut_bits = 5
        # --- offspring
        NewPopulation.clone_and_mutation_simple(n_parents=Nparents, n_clones=Nclones, mut_bits=mut_bits)
    elif pop < stage2_pop:
        # ##### STAGE TWO IN EVOLUTION ##### #
        # Description: high number of deaths, only cloning, small number of mutations
        # --- settings and killing
        kills = 40
        Nclones = kills
        NewPopulation.kill_chromes(n_murder=kills)  # killing worse
        Nparents = NewPopulation.parent_Nchromes  # number of parents
        mut_bits = 2
        # --- offspring
        NewPopulation.clone_and_mutation_simple(n_parents=Nparents, n_clones=Nclones, mut_bits=mut_bits)
    else:
        # ##### STAGE THREE IN EVOLUTION ##### #
        # Description: high number of deaths, cloning and reproduction, small number of mutations
        # --- settings and killing
        kills = 40
        NewPopulation.kill_chromes(n_murder=kills)  # killing worse
        # clone
        Nclones = 6  # how many clones we have to make
        Nparent_clones = 2  # parents, which is going to clone
        mut_bits = 2
        # reproduction
        Nchilds = kills - Nclones  # how many children chromosomes we will have
        # --- offspring
        # clone
        NewPopulation.clone_and_mutation_simple(n_parents=Nparent_clones, n_clones=Nclones, mut_bits=mut_bits)
        # reproduction
        NewPopulation.reproduction_fitvalue(child_chrom=Nchilds)

    # --- print survive: future parent
    list_print = NewPopulation.return_fitvalue_parent()
    print("After killing: fitValues (future parent)")
    pprint.pprint(list_print)

    # --------------------------- after offspring --------------------------- #
    # get new chromosomes
    new_chromes_map = NewPopulation.return_chromes_map()
    # update in `ChessboardPattern`
    ChessEvolution.update_chromosomes_map(new_chromes_map)

    # --------------------------- PLOT: LATTICE --------------------------- #
    new_lattice = ChessEvolution.return_lattice(index_lattice=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(new_lattice, interpolation='nearest', cmap='viridis')
    cax.set_clim(vmin=0, vmax=1)
    cbar = fig.colorbar(cax, ticks=[0, 0.3, 0.5, 1], orientation='vertical')
    # plt.show()
    napis = f'{pop:03d}'

    plt.savefig(output_plots_path + '/Chess-in-population-' + napis + '.png', dpi=300)
    plt.close(fig)

    # --------------------------- LAST STEP --------------------------- #
    if pop == NumsOfPops - 1:
        # firstly we randomise initial lattices
        ChessEvolution.update_initial_lattice(n_size=n_size, init_lattices=5)
        # do a population step
        ChessEvolution.one_population(time_steps=100, steps_to_average=6)
        # after that return chromosomes map
        chromes_map = ChessEvolution.return_chromosomes_map()
        # last chromosomes
        last_chromes_map = chromes_map

# ------------------------------- MAKE GIF ------------------------------- #
make_gif(output_plots_path)
# ------------------------------- PLOT: FITTING ------------------------------- #
fig, ax = plt.subplots(figsize=(9.0, 6.0))
ax.set_title(f'Fitting function vs time', fontsize=22, loc='left')
# data
ax.plot(pop_list, fitValue_part_list, label=r"Evolution")
# description
ax.set_ylabel(r"fitValue", fontsize=18)
ax.set_xlabel(r"Populations", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend()
# plt.show()
plt.savefig(output_plots_path + '/Fitting-function-vs-time.png')
plt.close(fig)

# ------------------------------- SAVE FILE .TXT ------------------------------- #
Nelem = len(last_chromes_map)
index_list = np.array([])
items_str_list = []
items_fitValue_list = np.array([])

for i in range(0, Nelem):
    index_list = np.append(index_list, i)
    # fitValue
    fitValue = last_chromes_map[i]["fitValue"]
    items_fitValue_list = np.append(items_fitValue_list, fitValue)
    # binaryStr
    binaryStr = list(last_chromes_map[i]["binaryStr"])
    print("binaryStr", binaryStr)
    items_str_list.append(binaryStr)

data_txt = []
for i in range(0, Nelem):
    data_txt.append([index_list[i], items_fitValue_list[i], items_str_list[i]])
print(data_txt)
file_path = output_plots_path + '/data.csv'

with open(file_path, mode="w", newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerows(data_txt)
