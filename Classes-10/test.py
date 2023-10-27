import numpy as np
from pprint import pprint


N = 120
zeros_1D = np.zeros(N)

Nx = 15
Ny = 8

zeros_2D = np.asarray(zeros_1D).reshape(Ny, Nx)

pprint(zeros_2D)

zeros_2D[0, :] = 1
zeros_2D[-1, :] = 1
print("------------------------------------------------------------")
pprint(zeros_2D)

print("------------------------------------------------------------")
g_ip1_j = np.roll(np.roll(zeros_2D, +1, axis=0), 0, axis=1)  # 7
pprint(g_ip1_j)

# print("------------------------------------------------------------")
# pprint(zeros_2D.flatten())


lista_1 = [0, 0, 0, 0]
lista_2 = [1, 1, 1, 1]
lista_3 = [2, 2, 2, 2]

max_list = max(lista_1, lista_2, lista_3)
print(max_list)

print("------------------------------------------------------------")
# Example 2D arrays
array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[2, 4, 6], [8, 10, 12]])
array3 = np.array([[5, 5, 5], [5, 5, 5]])
array4 = np.array([[3, 1, 9], [7, 3, 2]])

# Combine all arrays into a 3D array
combined_array = np.stack((array1, array2, array3, array4))

# Find the maximum value at each index along the axis 0
result_array = np.max(combined_array, axis=0)

print(result_array)


print("------------------------------------------------------------")
def compare_arrays(array1, array2):
    np_array1 = np.array(array1)
    np_array2 = np.array(array2)

    if np_array1.shape != np_array2.shape:
        return True  # Arrays have different dimensions

    return not np.array_equal(np_array1, np_array2)

# Example arrays
array1 = [[1, 2, 3], [4, 5, 6]]
array2 = [[1, 2, 3], [4, 5, 7]]

if compare_arrays(array1, array2):
    print("Arrays are different.")
else:
    print("Arrays are identical.")

print(compare_arrays(array1, array2))