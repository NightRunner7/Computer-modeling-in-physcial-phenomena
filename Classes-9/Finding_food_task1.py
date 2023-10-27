import numpy as np
from Ants_and_nest import AntClass, SpaceWithAnts
import matplotlib.pyplot as plt
from pprint import pprint
import os
import glob
from PIL import Image

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

def do_color_coding(grid_simple):
    """
    return:
    """
    cell_N = 1.0 * (np.where(grid_simple == 'N', 1, 0)).astype(int)
    cell_O = 2.0 * (np.where(grid_simple == 'O', 1, 0)).astype(int)
    cell_F = 3.0 * (np.where(grid_simple == 'F', 1, 0)).astype(int)
    cell_B = 4.0 * (np.where(grid_simple == 'B', 1, 0)).astype(int)

    state = cell_N + cell_O + cell_F + cell_B
    return state.copy()

# ---------------------------------- PREPARE GRID ---------------------------------- #
n_size = 80  # size of grid
N_pos_x = 50  # pos_x of nest
N_pos_y = 50  # pos_y of nest
F_max_y = 70  # big chung of food max y-axis
F_min_y = 10  # big chung of food min y-axis
F_max_x = 30  # big chung of food max y-axis
F_min_x = 5  # big chung of food min y-axis

def prepare_grid(size, N_pos_x, N_pos_y, F_x, F_y):
    """
    :param size:
    :param N_pos_x:
    :param N_pos_y:
    :param F_x:
    :param F_y:
    :return:
    """
    grid = np.array([x[:] for x in [[None] * size] * size])
    # the nest
    grid[N_pos_x][N_pos_y] = 'N'
    # localization the food
    for i in range(F_x[0], F_x[1]):
        for j in range(F_y[0], F_y[1]):
            grid[i][j] = 'F'

    # simple cells
    grid = np.where(grid == None, 'O', grid)

    return grid.T

init_grid = prepare_grid(n_size, N_pos_x, N_pos_y, [F_min_x, F_max_x], [F_min_y, F_max_y])
# pprint(init_grid)

# --- check initial grid
after_color_coding = do_color_coding(init_grid)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(after_color_coding, interpolation='nearest', cmap='viridis')
cax.set_clim(vmin=1, vmax=4)
cbar = fig.colorbar(cax, ticks=[1.0, 2.0, 3.0, 4.0], orientation='vertical')
# plt.show()

plt.savefig('./Initial-word.png', dpi=300)
plt.close(fig)

# ---------------------------------- SET PARAMETERS ---------------------------------- #
# paths
output_path_food = "./Task1-vol2-pheromones-food"
output_path_home = "./Task1-vol2-pheromones-home"
output_path_all = "./Task1-vol2-pheromones-all"
make_directory(output_path_food)
make_directory(output_path_home)
make_directory(output_path_all)
# parameters
nest_loc = [N_pos_x, N_pos_y]
nmax_ants = 200
alpha = 5
const_h = 2
beta = 0.99

# --- create class
WordClass = SpaceWithAnts(init_grid, nest_loc, nmax_ants=nmax_ants, alpha=alpha, const_h=const_h, beta=beta)

# ---------------------------------- ITERATIONS ---------------------------------- #
steps = 1000

for step in range(0, steps):

    print("I m doing iteration:", step)

    # do one iteration
    WordClass.one_iteration()

    # --- evolution food pheromones
    if step % 10 == 0:
        food_pheromones = WordClass.return_food_pheromones()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax1 = ax.imshow(food_pheromones, cmap='hot')
        # cax.set_clim(vmin=1, vmax=4)
        # cbar = fig.colorbar(cax, ticks=[1.0, 2.0, 3.0, 4.0], orientation='vertical')
        # plt.show()
        napis = f'{step:04d}'
        plt.savefig(output_path_food + '/food-pheromones-in-iteration-' + napis + '.png')
        plt.close(fig)

        # --- evolution food pheromones
        home_pheromones = WordClass.return_home_pheromones()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax2 = ax.imshow(home_pheromones, cmap='hot')

        napis = f'{step:04d}'
        plt.savefig(output_path_home + '/home-pheromones-in-iteration-' + napis + '.png')
        plt.close(fig)

        # --- evolution food pheromones
        all_pheromones = WordClass.return_all_pheromones()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax3 = ax.imshow(all_pheromones, cmap='hot')

        napis = f'{step:04d}'
        plt.savefig(output_path_all + '/all-pheromones-in-iteration-' + napis + '.png')
        plt.close(fig)

# --- make gif
make_gif(output_path_food, "food-pheromones-evolution")
make_gif(output_path_home, "home-pheromones-evolution")
make_gif(output_path_all, "all-pheromones-evolution")

non_zeo_beta = 0
for i in range(0, 80):
    beta = WordClass.Ants[i].beta
    if beta > 0:
        non_zeo_beta += 1

print("How many ants do not get lost:", non_zeo_beta)
print("How many food in the nest in the end:", WordClass.food_in_nest)
