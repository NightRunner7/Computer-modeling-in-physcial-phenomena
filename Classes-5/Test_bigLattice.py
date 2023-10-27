import sys
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
# ------------------------------- GET CHROMOSOMES -------------------------------#
output_plots_path = currentdir + '/big_lattice'
chromes_file = currentdir + '\\big_lattice' + '\\data_best3.csv'
# chromes str
chromes_str_list = []
number_3 = 50
with open(chromes_file, 'r') as file:
    csvreader = csv.reader(file, delimiter='\t')
    for row in csvreader:
        chromes_str = np.array(row[2].strip('][').split(', '), dtype=int)
        if int(float(row[0])) < number_3:
            chromes_str_list.append(chromes_str)

number_1 = 0
chromes_file = currentdir + '\\big_lattice' + '\\data_best1.csv'
with open(chromes_file, 'r') as file:
    csvreader = csv.reader(file, delimiter='\t')
    for row in csvreader:
        chromes_str = np.array(row[2].strip('][').split(', '), dtype=int)
        if int(float(row[0])) < number_1:
            chromes_str_list.append(chromes_str)

number_2 = 0
chromes_file = currentdir + '\\big_lattice' + '\\data_best2.csv'
with open(chromes_file, 'r') as file:
    csvreader = csv.reader(file, delimiter='\t')
    for row in csvreader:
        chromes_str = np.array(row[2].strip('][').split(', '), dtype=int)
        if int(float(row[0])) < number_2:
            chromes_str_list.append(chromes_str)

# number_4 = 10
# chromes_file = currentdir + '\\big_lattice' + '\\data_best1_vol2.csv'
# with open(chromes_file, 'r') as file:
#     csvreader = csv.reader(file, delimiter='\t')
#     for row in csvreader:
#         chromes_str = np.array(row[2].strip('][').split(', '), dtype=int)
#         if int(float(row[0])) < number_4:
#             chromes_str_list.append(chromes_str)

# ------------------------------- GET CHROMOSOMES -------------------------------#
# --- create map
Nchrom = len(chromes_str_list)
chromes_map = {}
for chrom in range(Nchrom):
    # random chromosome `string`
    chrom_str = chromes_str_list[chrom]
    # chromosome attributes
    chromes_atr = {"binaryStr": chrom_str, "fitValue": None}
    chromes_map[chrom] = chromes_atr
# --- create class
init_lattices = 1
n_size = 500
ChessEvolution = ChessboardPattern(n_size=n_size, init_lattices=init_lattices, nchrom=Nchrom)
# --- update chromosomes
ChessEvolution.update_chromosomes_map(chromes_map)
# print(chromes_map)

#

# save best one
data_txt = [chromes_str_list[0]]
print(len(data_txt[0]))
file_path = output_plots_path + "\\KrzysztofSzafranski_chromosome.csv"
with open(file_path, mode="w", newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerows(data_txt)
