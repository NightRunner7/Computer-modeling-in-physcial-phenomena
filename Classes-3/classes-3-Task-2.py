import numpy as np
from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt

#------------------------------- HERMITIAN MATRICES -------------------------------#
def Hamiltonian_matrix_GOE(size):
    """
    Set and (randomise) the quadratic matrix. So we want to firstly
    randomise NxN matrix (size refers to N). Then we take that
    our matrix is hermitian.

        H_{ij} is element of the matrix, which is real

    ---------
    :param: size, refers to size of matrix (size = N).
    :return: hermitian matrix (NxN)

    GOE: devotion to gaussian orthogonal ensemble.
    """
    h = np.random.randn(size, size)  # randomise the H_{ij} elements
    return (h + h.T) / 2  # take care that our H_{NxN} matrix is hermitian


def Hamiltonian_matrix_GUE(size):
    """
    Set and (randomise) the quadratic matrix. So we want to firstly
    randomise NxN matrix (size refers to N). Then we take that
    our matrix is hermitian.

        H_{ij} is element of the matrix, which is complex.
        z = x + y*i
        Re(z) = x, Im(z) = y

        H_sys = (H + H_dagger) / 2,
    where:
        H_sys: hermitian hamiltonian of a system: NxN - size.
        H: hamiltonian, where H_{ij} is complex number.
        H_dagger = (H)^(dagger) is a Hermitian conjugate of a matrix H.
    ---------
    :param: size, refers to size of matrix (size = N).
    :return: hermitian matrix (NxN)

    GUE: devotion to gaussian unitary ensemble
    """
    x = np.random.randn(size, size)  # randomise Re(z)
    y = np.random.randn(size, size)  # randomise IM(z)
    h = (x + 1j * y) / np.sqrt(2)  # set the
    h_dagger = (x.T - 1j * y.T) / np.sqrt(2)
    return (h + h_dagger) / 2


def Find_eigenValues(matrix):
    """
    Find the eigenvalues of matrix. In our cases, the physical meaning
    of that function is just find the Eigenstates of energy - because
    the input matrix is hamiltonian matrices.

    :param matrix: is a hermitian matrix.
    :return: an eigenvalues for input matrix.
    """
    eigenVal = eigvalsh(matrix)  # finding the eigenvalues
    return eigenVal

"""
little testing: that we finding the eigenvalues in GUE case
"""

Gue_matrix = Hamiltonian_matrix_GUE(2)
print(Gue_matrix)

Gue_matrix_eigenVals = Find_eigenValues(Gue_matrix)
print('Eigenvalues of complex matrix:', Gue_matrix_eigenVals)

#------------------------------- NECESSARY fUNCTIONS -------------------------------#
def Wigner_GOE(_s):
    """
    GOE, refers to gaussian orthogonal ensemble.

    :param _s: the normalising spacing of eigne energies.
    :return: the density of the probability of that spacing in the system.
    """
    return np.pi / 2 * _s * np.exp(- np.pi / 4 * _s**2)

def Wigner_GUE(_s):
    """
    GUE, refers to gaussian unitary enesamble.

    :param _s: the normalising spacing of eigne energies.
    :return:
    """
    return 32 / np.pi**2 * _s**2 * np.exp(- 4 / np.pi * _s**2)


def cal_average(_list):
    average = 0
    Nums = len(_list)
    for i in range(0, Nums):
        average = average + _list[i] / Nums

    return average

def normalised_spacing(difference_list):
    """
    Find the list of the normalised energy spacing.

    :param energy_list: the list contains (the sorted) values of state energy.
    :return: list of normalised energy spacing.
    """
    # list containg differences between elements, so now we have spacing
    # diff = abs(np.diff(energy_list))
    # compute average
    N = len(difference_list)
    avr_spacing = cal_average(difference_list)

    nor_spacing = np.array([])
    for i in range(0, N):
        spacing = 1/avr_spacing * difference_list[i]  # normalised spacing
        # append
        nor_spacing = np.append(nor_spacing, spacing)

    return nor_spacing

def selection(list_eVals):
    """
    :param list_eigenVals:
    :return:
    """
    N = len(list_eVals)

    if N < 3:
        left_arg = int(N / 2 - 1) - 1
        right_arg = int(N / 2) + 1

        sel_eVals = np.array(list_eVals[left_arg:right_arg])
        return sel_eVals
    else:
        elements = int(N / 4)
        mid_element = int(N / 2)
        left_arg = int(mid_element - elements / 2)
        right_arg = int(mid_element + elements/ 2)

        sel_eVals = np.array(list_eVals[left_arg:right_arg])
        return sel_eVals

#------------------------------- FINDING: NORMALISED SPACING-------------------------------#
def spacind_normalised(_N, _nSamps):
    whole_spacing_goe = np.array([])
    whole_spacing_gue = np.array([])

    for i in range(0, _nSamps):
        # randomise hamiltonian matrices
        h_goe = Hamiltonian_matrix_GOE(_N)
        h_gue = Hamiltonian_matrix_GUE(_N)
        # find eigenvalues
        eigenVals_goe = Find_eigenValues(h_goe)
        eigenVals_gue = Find_eigenValues(h_gue)
        # sorting
        eigenVals_goe = np.sort(eigenVals_goe)
        eigenVals_gue = np.sort(eigenVals_gue)
        # selection
        sel_eVals_goe = selection(eigenVals_goe)
        sel_eVals_gue = selection(eigenVals_gue)


        # calculade differences
        if _N > 10:
            sel_diff_goe = np.diff(sel_eVals_goe)
            sel_diff_gue = np.diff(sel_eVals_gue)
        else:
            sel_diff_goe = np.array(sel_eVals_goe)
            sel_diff_gue = np.array(sel_eVals_gue)

        '''
        give me normalised spacing
        '''
        if _N > 10:
            nor_spacing_goe = normalised_spacing(sel_diff_goe)  # GOE
            nor_spacing_gue = normalised_spacing(sel_diff_gue)  # GUE
        else:
            sel_eVals_goe = np.sort(sel_eVals_goe)
            nor_spacing_goe = -sel_eVals_goe[0] + sel_eVals_goe[-1]

            sel_eVals_gue = np.sort(sel_eVals_gue)
            nor_spacing_gue = -sel_eVals_gue[0] + sel_eVals_gue[-1]


        # appending
        whole_spacing_goe = np.append(whole_spacing_goe, nor_spacing_goe)
        whole_spacing_gue = np.append(whole_spacing_gue, nor_spacing_gue)

        # eVals_goe = np.append(eVals_goe, eigenVals_goe)
        # eVals_gue = np.append(eVals_gue, eigenVals_gue)


    return [whole_spacing_goe, whole_spacing_gue]


N = 8 # N = 20, 200
nSamps = 20000 # nSamps = 10000, 500
# finding spacing
nor_spacing_goe, nor_spacing_gue = spacind_normalised(N,nSamps)
# print('nor_spacing_goe', nor_spacing_goe)
# print('nor_spacing_gue', nor_spacing_gue)

# eVals_goe = np.sort(eVals_goe)
# eVals_gue = np.sort(eVals_gue)
print('len(nor_spacing_goe)', len(nor_spacing_goe))
print('len(nor_spacing_gue)', len(nor_spacing_gue))

"""
Plotting histogram
"""
def normalise_list(list):
    N = len(list)
    average = cal_average(list)
    normalise_list = np.array([])
    for i in range(0,N):
        normalise_val = 1/average * list[i]
        normalise_list = np.append(normalise_list, normalise_val)

    return normalise_list

nor_data_goe = normalise_list(nor_spacing_goe)

fig, ax = plt.subplots(figsize=(9.0, 6.0))
# data
n, bins, cos = plt.hist(nor_data_goe, 50, density=True, facecolor='cyan', alpha=0.75)
plt.plot(bins, Wigner_GOE(bins), 'r-', linewidth=2)
# describing
plt.title(r'Frequency of occurrence spacing (GOE)', fontsize=20)
plt.ylabel('Normalised occurrence', fontsize=20)
plt.xlabel(r'Energy spacing: $s$', fontsize=20)
# plt.show()
plt.savefig('Frequency_occurrence_spacing_GOE_N_' + str(N) + '_Nsample_' + str(nSamps) + '.png')
plt.close(fig)



"""
Plotting histogram
"""

nor_data_gue = normalise_list(nor_spacing_gue)


fig, ax = plt.subplots(figsize=(9.0, 6.0))
# data
n, bins, cos = plt.hist(nor_data_gue, 50, density=True, facecolor='cyan', alpha=0.75)
plt.plot(bins, Wigner_GUE(bins), 'r-', linewidth=2)
# describing
plt.title(r'Frequency of occurrence spacing (GUE)', fontsize=20)
plt.ylabel('Normalised occurrence', fontsize=20)
plt.xlabel(r'Energy spacing: $s$', fontsize=20)
# plt.show()
plt.savefig('Frequency_occurrence_spacing_GUE_N_' + str(N) + '_Nsample_' + str(nSamps) + '.png')
plt.close(fig)