import numpy as np
import sys
from FishAndSharp import FishClass, SharkClass, WaTorUniverse
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.signal import find_peaks

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
                   save_all=True, duration=50, loop=0)


# -------------------------------- INITIALIZATION -------------------------------- #
# --- parameters
n_size = 200  # 40, 200
n_fish = 300*25  # 300, 300*5*5
n_shark = 10*25  # 10, 10*5*5
fish_rep = 3
shark_rep = 20
hungry_lvl = 3
# --- output
output_path_food = f"Task1_Nsize_{n_size}_Nshark_{n_shark}_Nfish_{n_fish}_folder_2"
make_directory(output_path_food)

# --- class initialization
OceanClass = WaTorUniverse(n_size=n_size, n_fish=n_fish, n_shark=n_shark,
                           fish_rep=fish_rep, shark_rep=shark_rep, hungry_lvl=hungry_lvl)
# --- steps
N_steps = 800

sharp_population_list = np.array([])
fish_population_list = np.array([])
time_list = np.linspace(0, N_steps-1, N_steps)

# pprint(time_list)
# sys.exit()

# -------------------------------- LOOP OVER STEPS -------------------------------- #
for step in range(0, N_steps):
    print("I m doing step:", step)

    # --- update data
    sharp_population = OceanClass.n_shark
    sharp_population_list = np.append(sharp_population_list, sharp_population)
    fish_population = OceanClass.n_fish
    fish_population_list = np.append(fish_population_list, fish_population)

    if sharp_population == 0:
        sys.exit("No sharks, no need to do calculations")

    # --- We make the plot
    grid_color_coding = OceanClass.do_color_coding()

    fig = plt.figure()
    colors = ['blue', 'green', 'black']  # Define the colors for each value
    cmap = ListedColormap(colors)  # Create a custom color map
    plt.imshow(grid_color_coding, cmap=cmap, interpolation='nearest')  # Plot the array using the custom color map
    plt.colorbar()  # Add a color bar legend
    # plt.show()  # Display the plot
    napis = f'{step:04d}'
    plt.savefig(output_path_food + '/Ocean-sharks-and-fish-in-' + napis + '.png')
    plt.close(fig)

    # --- first we move and reproduce the fish.
    OceanClass.fish_move_and_reproduction()

    # --- secondly we move and reproduce the shark.
    OceanClass.shark_move_and_reproduction()

    # --- thirdly we will kill or starve the sharks.
    OceanClass.hungry_level_sharks()

    # --- fourthly we will update the time to reproduction.
    OceanClass.update_reproduction_time()

# --- making a gif
make_gif(output_path_food, "Evolution-of-Ocean")


# ----------------------------- PLOT THE RESULTS: ONE ----------------------------- #
# --- cutting
# peaks, _ = find_peaks(fish_population_list, distance=24)  # good for `n_size = 40`
peaks, _ = find_peaks(fish_population_list, distance=77)
nelements = 7
# difference between peaks is >= 150
print("minimas:", np.diff(peaks))
print("peaks:", peaks)


fig, ax = plt.subplots(1, 1, figsize=(12.0, 8.0))
ax.set_title('Changing size of population in time', fontsize=18)

# --- population
ax.plot(time_list, sharp_population_list, label=r"sharks population")
ax.plot(time_list, fish_population_list, label=r'fish population')
plt.plot(peaks, fish_population_list[peaks], "x", label=r"peaks, which we mark")


# describe plot
ax.set_xlabel(r"$t$ [steps]", fontsize=18)
ax.set_ylabel(r"size of population", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True)
# ax.legend
ax.legend()
plt.savefig(f'./Changing-in-population-Nsize-{n_size}-Nshark-{n_shark}-Nfish-{n_fish}.png')
# plt.show()
plt.close(fig)

# ----------------------------- PLOT THE RESULTS: TWO ----------------------------- #
# --- select the period of peaks: oscillations
beg_one_index = peaks[0]
end_one_index = peaks[1]

beg_two_index = peaks[1]
end_two_index = peaks[2]

# beg_three_index = peaks[4]
# end_three_index = peaks[5]

# beg_index = 0
# end_index = 120
sharp_one_period = sharp_population_list[beg_one_index:(end_one_index + nelements)]
fish_one_period = fish_population_list[beg_one_index:(end_one_index + nelements)]

sharp_two_period = sharp_population_list[beg_two_index:(end_two_index + nelements)]
fish_two_period = fish_population_list[beg_two_index:(end_two_index + nelements)]

# sharp_three_period = sharp_population_list[beg_three_index:(end_three_index + nelements)]
# fish_three_period = fish_population_list[beg_three_index:(end_three_index + nelements)]

# --- JUST PLOT
fig, ax = plt.subplots(1, 1, figsize=(10.0, 10.0))
ax.set_title('Phase trajectories of sharks and fish', fontsize=18)

# --- population
# ax.plot(sharp_population_list, fish_population_list, label=r"sharp's population")
ax.plot(sharp_one_period, fish_one_period, label=r"first period")
ax.plot(sharp_two_period, fish_two_period, label=r"second period")
# ax.plot(sharp_three_period, fish_three_period, label=r"third period")

# describe plot
ax.set_xlabel(r"population of sharks", fontsize=18)
ax.set_ylabel(r"population of fish", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True)
# ax.legend
ax.legend()
plt.savefig(f'./Phase-trajectories-Nsize-{n_size}-Nshark-{n_shark}-Nfish-{n_fish}.png')
# plt.show()
plt.close(fig)
