import numpy as np
from scipy.fft import fft, ifft

# Prepare normalized Gaussian wave packet: $\psi_{G} (x_{n})$
'''
We will prepare the normalised Gaussian wave packet. We will use following shortcuts:
    cal: calculate
    MoI: methof of images
    norm: normalisation
    f: factor

In this cell we just set some of functions, which we will use to find normalised Gaussian
wave packet
'''


################################ AMPLITUDE AND NORMALISATION ################################
def cal_amplitude(_wave):
    """
    Calculate the amplitude of taken single _wave function

    :param: _wave: single wave function [complex vector].
    return: the amplitude of _wave function [float]
    """
    amplitude = _wave.real * _wave.real + _wave.imag * _wave.imag
    return amplitude


def cal_norm(_sup_waves):
    """
    Find the normalisation of the superpositions of wave functions

    :param: _sup_waves: list of wave functions, which create superpositions wave function.
                        In different word we take into account every wave function, which
                        exists in system.
    return: normalisation factor
    """
    n_waves = len(_sup_waves)  # number of waves
    f_norm = 0

    for i in range(0, n_waves):
        amplitude = cal_amplitude(_sup_waves[i])
        f_norm = f_norm + np.sqrt(amplitude)

    return f_norm


################################ SINGLE WAVE FUNCTION ################################
def psi0(x, x0, p0, heff):
    """
    :param: x: descritised position
    return: single wave function [complex vector].
    """
    val_psi0 = np.exp(1j * p0 * x / heff) * np.exp(- (x - x0) ** 2 / (2 * heff))
    return val_psi0


def cal_wave_MoI(x, x0, p0, heff):
    """
    take linear combination of some wave function: 'method of images'
    that gives us wave function, which is periodic.

    :param: x: descritised position
    return: superposition of few wave functions [complex vector]
    """
    # we will sum up for |d| = 4.
    val_psiG = 0
    for d in range(-4, 4):
        val_psiG = val_psiG + psi0(x + 2 * np.pi * d, x0, p0, heff)

    return val_psiG


def normGaussianPacked(_x_list, x0, p0, heff):
    """
    Find the normalised gaussian wave function

    :param: _x_list: list containg the values of position
    return: normalised gaussian wave factor list of [complex vector]
    """
    M = len(_x_list)
    psiG = np.zeros(M, dtype=complex)  # complex vector

    # calculate gaussian wave package
    for i in range(0, M):
        x = _x_list[i]
        # calculate gaussian packaked
        val_psiG = cal_wave_MoI(x, x0, p0, heff)
        psiG[i] = val_psiG

    # find normalised factor
    f_norm = cal_norm(psiG)

    # rescalled gaussian wave packed: Normalisation
    for i in range(0, M):
        psiG[i] = 1 / f_norm * psiG[i]

    return psiG

# Do one step in gaussian packet evolution
"""
We calculate the quantum dynamics of our standard map. In another words
this time our gaussian wave packed will evolves.
"""


def pot_V(x, _K, heff):
    """
    The phase vector, which depends on position. Phase vector for
    in the positional notation / space.

    :param: x: descritised position.
    :param: _K: some constant.
    return: Phase vector [complex vector]
    """
    val_V = np.exp(-1j / heff * _K * np.cos(x))
    return val_V


def Momenta_P(p, heff):
    """
    The phase vector, which depends on momentum. Phase vector for
    in the momentum notation / space.

    :param: p: descritised momentum.
    return: Phase vector [complex vector]
    """
    val_P = np.exp(-1j / (2 * heff) * p ** 2)
    return val_P


def do_one_step(_wavef, _K, x_list, p_list, heff):
    """
    Do one step in evolution our _wavef.

    :param: _wavef: normalised gaussian wave function
    :param: _K: some constant. Using for pot_V(x, _K).
    return: gaussian packed wave fanctions. List of [complex vector] after a 'one step'
            in evolution. One remark: after this evolution it will no longer be normalised!
    """
    psiG_evo = []
    M = len(x_list)
    # first step
    for i in range(0, M):
        x = x_list[i]
        # calculate
        _wavef[i] = _wavef[i] * pot_V(x, _K, heff)

    # second step
    fft_psiG = fft(_wavef)

    # third step
    for i in range(0, M):
        p = p_list[i]
        # calculate
        fft_psiG[i] = fft_psiG[i] * Momenta_P(p, heff)

    # fourth step
    psiG_evo = ifft(fft_psiG)
    return psiG_evo