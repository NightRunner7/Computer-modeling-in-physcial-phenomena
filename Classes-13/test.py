import numpy as np
import random
from pprint import pprint


my_list = ["W", "W", "F", "S", "F", "F", "W", "W"]
my_list_2 = ["W", "W"]


# Find indices of all occurrences of "F"
indices = [index for index, value in enumerate(my_list) if value == "F"]

# Select a random index from the list of indices
random_index = random.choice(indices)

# Retrieve the selected "F" element and its index
selected_element = my_list[random_index]

print("Selected element:", selected_element)
print("Index:", random_index)


# Find indices of all occurrences of "F"
indices = [index for index, value in enumerate(my_list_2) if value == "F"]

# Select a random index from the list of indices
if indices:
    random_index = random.choice(indices)
    print("indices", indices)
    print("random_index", random_index)
else:
    print("indices", indices)
    print("empty list")


fish_id = np.arange(1, 10*10, 1, dtype=int)
sharp_id = np.arange(-1, -10*10, -1, dtype=int)
print("id fish")
pprint(fish_id)
print("id sharp")
pprint(sharp_id)

fish_id = fish_id[1:]
print("id fish: after minus 1 element")
pprint(fish_id)


# ---
# Dictionary with two keys
Dictionary1 = {'A': 'Geeks', 'B': 'For'}

# Printing keys of dictionary
print("Keys before Dictionary Updation:")
keys = list(Dictionary1.keys())
print(keys)
print(keys[0])


# ---
print("radint:", random.randint(0, 3))

