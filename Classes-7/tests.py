import numpy as np
from pprint import pprint

names_1 = [['a', 'b', 'c'],
           ['d', 'e', 'f'],
           ['g', 'h', 'i']]

names_2 = [['1', '2', '3'],
           ['4', '5', '6'],
           ['7', '8', '9']]

# self.color_coding = np.array([[None for x in range(self.n_size)] for y in range(self.n_size)])

names_3 = [[names_1[i][j] + names_2[i][j] for i in range(len(names_1))] for j in range(len(names_1[0]))]

print(names_3)

names_4 = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]])

print("sum:", np.sum(names_4))

print(type(names_3))

names_join = np.array([names_1, names_2])
pprint(names_join)

n_size = 6
beh_grid_previous = np.array(5 * [[[None] * n_size] * n_size])

print(beh_grid_previous)

try_lisy = np.array(names_1)
pprint(try_lisy)


initial_beh = np.where(try_lisy == 'a', 'D' * 9, 'C' * 9)
pprint(initial_beh)


