from __future__ import annotations

import numpy as np

from colour.algebra import sdiv, sdiv_mode, spow, vector_dot
from colour.hints import ArrayLike, NDArrayFloat, Tuple
from colour.utilities import (
    as_float,
    as_float_array,
    tsplit,
    tstack,
)

# Code in this module has been lifted and adapted from Colour Science library

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"


def degree_of_adaptation(F: ArrayLike, L_A: ArrayLike) -> NDArrayFloat:
    """
    Parameters
    ----------
    F
        Surround maximum degree of adaptation :math:`F`.
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.
    """

    F = as_float_array(F)
    L_A = as_float_array(L_A)

    D = F * (1 - (1 / 3.6) * np.exp((-L_A - 42) / 92))

    return D


def viewing_conditions_dependent_parameters(
    Y_b: ArrayLike,
    Y_w: ArrayLike,
    L_A: ArrayLike,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Parameters
    ----------
    Y_b
        Adapting field *Y* tristimulus value :math:`Y_b`.
    Y_w
        Whitepoint *Y* tristimulus value :math:`Y_w`.
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.
    """

    Y_b = as_float_array(Y_b)
    Y_w = as_float_array(Y_w)

    with sdiv_mode():
        n = sdiv(Y_b, Y_w)

    F_L = luminance_level_adaptation_factor(L_A)
    z = base_exponential_non_linearity(n)

    return F_L, z


def luminance_level_adaptation_factor(
    L_A: ArrayLike,
) -> NDArrayFloat:
    """
    Parameters
    ----------
    L_A
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`.
    """

    L_A = as_float_array(L_A)

    k = 1 / (5 * L_A + 1)
    k4 = k**4
    F_L = 0.2 * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * spow(5 * L_A, 1 / 3)

    return as_float(F_L)


def base_exponential_non_linearity(
    n: ArrayLike,
) -> NDArrayFloat:
    """
    Parameters
    ----------
    n
        Function of the luminance factor of the background :math:`n`.
    """

    n = as_float_array(n)

    z = 1.48 + np.sqrt(n)

    return z


def post_adaptation_non_linear_response_compression_forward(RGB, F_L):
    F_L_RGB = spow(F_L * np.abs(RGB) / 100.0, 0.42)
    RGB_c = (400.0 * np.sign(RGB) * F_L_RGB) / (27.13 + F_L_RGB)
    return RGB_c


def post_adaptation_non_linear_response_compression_inverse(
    RGB: ArrayLike, F_L: ArrayLike
) -> NDArrayFloat:
    """
    Parameters
    ----------
    RGB
        *CMCCAT2000* transform sharpened *RGB* array.
    F_L
        *Luminance* level adaptation factor :math:`F_L`.
    """

    RGB = as_float_array(RGB)
    F_L = as_float_array(F_L)

    RGB_p = (
        np.sign(RGB)
        * 100
        / F_L[..., None]
        * spow(
            (27.13 * np.absolute(RGB)) / (400 - np.absolute(RGB)),
            1 / 0.42,
        )
    )

    return RGB_p


def matrix_post_adaptation_non_linear_response_compression(
    P_2: ArrayLike, a: ArrayLike, b: ArrayLike
) -> NDArrayFloat:
    P_2 = as_float_array(P_2)
    a = as_float_array(a)
    b = as_float_array(b)

    RGB_a = (
        vector_dot(
            [
                [460, 451, 288],
                [460, -891, -261],
                [460, -220, -6300],
            ],
            tstack([P_2, a, b]),
        )
        / 1403
    )

    return RGB_a


def achromatic_response_forward(RGB: ArrayLike) -> NDArrayFloat:
    R, G, B = tsplit(RGB)
    A = 2 * R + G + 0.05 * B
    return A


def achromatic_response_inverse(
    A_w: ArrayLike,
    J: ArrayLike,
    c: ArrayLike,
    z: ArrayLike,
) -> NDArrayFloat:
    A_w = as_float_array(A_w)
    J = as_float_array(J)
    c = as_float_array(c)
    z = as_float_array(z)

    A = A_w * spow(J / 100, 1 / (c * z))

    return A


def opponent_colour_dimensions_forward(RGB: ArrayLike) -> NDArrayFloat:
    """
    Parameters
    ----------
    RGB
        Compressed *CMCCAT2000* transform sharpened *RGB* array.
    """

    R, G, B = tsplit(RGB)

    a = R - 12 * G / 11 + B / 11
    b = (R + G - 2 * B) / 9

    ab = tstack([a, b])

    return ab


def opponent_colour_dimensions_inverse(
    P_p_1: ArrayLike, h: ArrayLike, M: ArrayLike
) -> NDArrayFloat:
    """
    Parameters
    ----------
    P_p_1
        Point :math:`P'_1`.
    h
        Hue :math:`h` in degrees.
    M
        Correlate of *colourfulness* :math:`M`.
    """

    P_p_1 = as_float_array(P_p_1)
    M = as_float_array(M)

    hr = np.radians(h)

    with sdiv_mode():
        gamma = M / P_p_1

    a = gamma * np.cos(hr)
    b = gamma * np.sin(hr)

    ab = tstack([a, b])

    return ab


def hue_angle(a: ArrayLike, b: ArrayLike) -> NDArrayFloat:
    """
    Parameters
    ----------
    a
        Opponent colour dimension :math:`a`.
    b
        Opponent colour dimension :math:`b`.
    """

    a = as_float_array(a)
    b = as_float_array(b)

    h = np.degrees(np.arctan2(b, a)) % 360

    return as_float(h)


def lightness_correlate(
    A: ArrayLike,
    A_w: ArrayLike,
    c: ArrayLike,
    z: ArrayLike,
) -> NDArrayFloat:
    """
    Parameters
    ----------
    A
        Achromatic response :math:`A` for the stimulus.
    A_w
        Achromatic response :math:`A_w` for the whitepoint.
    c
        Surround exponential non-linearity :math:`c`.
    z
        Base exponential non-linearity :math:`z`.
    """

    A = as_float_array(A)
    A_w = as_float_array(A_w)
    c = as_float_array(c)
    z = as_float_array(z)

    with sdiv_mode():
        J = 100 * spow(sdiv(A, A_w), c * z)

    return J


def colourfulness_correlate(
    N_c: ArrayLike,
    e_t: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
) -> NDArrayFloat:
    """
    Parameters
    ----------
    N_c
        Surround chromatic induction factor :math:`N_{c}`.
    e_t
        Eccentricity factor :math:`e_t`.
    a
        Opponent colour dimension :math:`a`.
    b
        Opponent colour dimension :math:`b`.
    """

    N_c = as_float_array(N_c)
    e_t = as_float_array(e_t)
    a = as_float_array(a)
    b = as_float_array(b)

    M = 43.0 * N_c * e_t * np.hypot(a, b)

    return M


def P_p(
    N_c: ArrayLike,
    e_t: ArrayLike,
    A: ArrayLike,
) -> NDArrayFloat:
    """
    Parameters
    ----------
    N_c
        Surround chromatic induction factor :math:`N_{c}`.
    e_t
        Eccentricity factor :math:`e_t`.
    A
        Achromatic response  :math:`A` for the stimulus.
    """

    N_c = as_float_array(N_c)
    e_t = as_float_array(e_t)
    A = as_float_array(A)

    P_p_1 = 43 * N_c * e_t
    P_p_2 = A

    P_p_1 = np.ones(A.shape) * P_p_1
    P_p_n = tstack([P_p_1, P_p_2])

    return P_p_n
