import numpy as np
from pprint import pprint


def probability_set_direction(pheromones_list):
    """
    This function is used to choice the direction when ant
    have pheromones near her.

    :param pheromones_list:
    :return:

    `prob` denotes: probability
    `norm` denotes: normalization
    """
    # --- get normalization probability
    pherm_prob_list = np.array([])
    norm_prob = 0
    for i in range(0, len(pheromones_list)):
        prob = (11 + pheromones_list[i]) ** 5
        norm_prob += prob
        # appending
        pherm_prob_list = np.append(pherm_prob_list, prob)

    # do normalization
    for i in range(0, len(pherm_prob_list)):
        pherm_prob_list[i] = pherm_prob_list[i] / norm_prob

    print("pherm_prob_list:", pherm_prob_list)

    # --- get the index, which lead to the direction
    id_direction = np.random.choice(3, 1, replace=False, p=pherm_prob_list)[0]
    return id_direction

pheromones_list = [8, 4, 11]

for i in range(0, 10):
    index_direction = probability_set_direction(pheromones_list)
    print("index_direction:", index_direction)

# multiply
Ny = 10
Nx = 10
food_pheromones = np.array([x[:] for x in [[100] * Ny] * Nx])  # leave when searched for food
food_pheromones = food_pheromones * 0.99
pprint(food_pheromones)
