from __future__ import annotations

from collections import namedtuple

import numpy as np

from colour.algebra import sdiv, sdiv_mode, spow, vector_dot
from colour.hints import ArrayLike, NDArrayFloat, Tuple
from colour.utilities import (
    as_float,
    as_float_array,
    from_range_100,
    from_range_degrees,
    ones,
    tsplit,
    tstack,
)

from drt_cam_lib import (
    degree_of_adaptation,
    viewing_conditions_dependent_parameters,
    post_adaptation_non_linear_response_compression_forward,
    post_adaptation_non_linear_response_compression_inverse,
    matrix_post_adaptation_non_linear_response_compression,
    achromatic_response_forward,
    achromatic_response_inverse,
    compress_bjorn,
    uncompress_bjorn,
    opponent_colour_dimensions_forward,
    opponent_colour_dimensions_inverse,
    hue_angle,
    lightness_correlate,
    colourfulness_correlate,
    P_p,
)


# Code in this module has been lifted and adapted from Colour Science library

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"


CAT_CAT16: NDArrayFloat = np.array(
    [
        [0.401288, 0.650173, -0.051461],
        [-0.250268, 1.204414, 0.045854],
        [-0.002079, 0.048952, 0.953127],
    ]
)


class InductionFactors_Hellwig2022(
    namedtuple("InductionFactors_Hellwig2022", ("F", "c", "N_c"))
):
    pass


VIEWING_CONDITIONS_HELLWIG2022: dict = {
    "Average": InductionFactors_Hellwig2022(1, 0.69, 1),
    "Dim": InductionFactors_Hellwig2022(0.9, 0.59, 0.9),
    "Dark": InductionFactors_Hellwig2022(0.8, 0.525, 0.8),
}


def XYZ_to_JMh(
    XYZ: ArrayLike,
    XYZ_w: ArrayLike,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround: str = "Average",
    discount_illuminant: bool = False,
    matrix_lms: ArrayLike = CAT_CAT16,
    compress_mode: bool = False,
) -> NDArrayFloat:

    _X_w, Y_w, _Z_w = tsplit(XYZ_w)
    L_A = as_float_array(L_A)
    Y_b = as_float_array(Y_b)

    surround = VIEWING_CONDITIONS_HELLWIG2022[surround]

    # Step 0
    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    RGB_w = vector_dot(matrix_lms, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = (
        np.clip(degree_of_adaptation(surround.F, L_A), 0, 1)
        if not discount_illuminant
        else ones(L_A.shape)
    )

    F_L, z = viewing_conditions_dependent_parameters(Y_b, Y_w, L_A)

    D_RGB = D[..., None] * Y_w[..., None] / RGB_w + 1 - D[..., None]
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non-linear response compression.
    RGB_aw = post_adaptation_non_linear_response_compression_forward(RGB_wc, F_L)

    # Computing achromatic responses for the whitepoint.
    A_w = achromatic_response_forward(RGB_aw)

    # Step 1
    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    RGB = vector_dot(matrix_lms, XYZ)

    # Step 2
    RGB_c = D_RGB * RGB

    # Step 3
    # Applying forward post-adaptation non-linear response compression.

    if compress_mode:
        RGB_c = compress_bjorn(RGB_c)
        RGB_a = post_adaptation_non_linear_response_compression_forward(RGB_c, F_L)
        RGB_a = uncompress_bjorn(RGB_a)
    else:
        RGB_a = post_adaptation_non_linear_response_compression_forward(RGB_c, F_L)

    # Step 4
    # Converting to preliminary cartesian coordinates.
    a, b = tsplit(opponent_colour_dimensions_forward(RGB_a))

    # Computing the *hue* angle :math:`h`.
    h = hue_angle(a, b)

    # Step 5
    # Computing eccentricity factor *e_t*.
    e_t = 1.0

    # Step 6
    # Computing achromatic responses for the stimulus.
    A = achromatic_response_forward(RGB_a)

    # Step 7
    # Computing the correlate of *Lightness* :math:`J`.
    J = lightness_correlate(A, A_w, surround.c, z)

    # Step 9
    # Computing the correlate of *colourfulness* :math:`M`.
    M = colourfulness_correlate(surround.N_c, e_t, a, b)

    return tstack(
        [
            as_float(from_range_100(J)),
            as_float(from_range_100(M)),
            as_float(from_range_degrees(h)),
        ]
    )


def JMh_to_XYZ(
    JMh: ArrayLike,
    XYZ_w: ArrayLike,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround: str = "Average",
    discount_illuminant: bool = False,
    matrix_lms: ArrayLike = CAT_CAT16,
    compress_mode: bool = False,
) -> NDArrayFloat:

    J, M, h = tsplit(JMh)
    L_A = as_float_array(L_A)
    _X_w, Y_w, _Z_w = tsplit(XYZ_w)

    surround = VIEWING_CONDITIONS_HELLWIG2022[surround]

    # Step 0
    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    RGB_w = vector_dot(matrix_lms, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = (
        np.clip(degree_of_adaptation(surround.F, L_A), 0, 1)
        if not discount_illuminant
        else ones(L_A.shape)
    )

    F_L, z = viewing_conditions_dependent_parameters(Y_b, Y_w, L_A)

    D_RGB = D[..., None] * Y_w[..., None] / RGB_w + 1 - D[..., None]
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non-linear response compression.
    RGB_aw = post_adaptation_non_linear_response_compression_forward(RGB_wc, F_L)

    # Computing achromatic responses for the whitepoint.
    A_w = achromatic_response_forward(RGB_aw)

    # Step 2
    # Computing eccentricity factor *e_t*.
    e_t = 1.0

    # Computing achromatic response :math:`A` for the stimulus.
    A = achromatic_response_inverse(A_w, J, surround.c, z)

    # Computing *P_p_1* to *P_p_2*.
    P_p_n = P_p(surround.N_c, e_t, A)
    P_p_1, P_p_2 = tsplit(P_p_n)

    # Step 3
    # Computing opponent colour dimensions :math:`a` and :math:`b`.
    ab = opponent_colour_dimensions_inverse(P_p_1, h, M)
    a, b = tsplit(ab)

    # Step 4
    # Applying post-adaptation non-linear response compression matrix.
    RGB_a = matrix_post_adaptation_non_linear_response_compression(P_p_2, a, b)

    # Step 5
    # Applying inverse post-adaptation non-linear response compression.

    if compress_mode:
        RGB_a = compress_bjorn(RGB_a)
        RGB_c = post_adaptation_non_linear_response_compression_inverse(RGB_a, F_L)
        RGB_c = uncompress_bjorn(RGB_c)
    else:
        RGB_c = post_adaptation_non_linear_response_compression_inverse(RGB_a, F_L)

    # Step 6
    RGB = RGB_c / D_RGB

    # Step 7
    XYZ = vector_dot(np.linalg.inv(matrix_lms), RGB)

    return from_range_100(XYZ)


def JMh_to_luminance_RGB(
    JMh: ArrayLike,
    XYZ_w: ArrayLike,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround,
    discount_illuminant,
    matrix_lms,
    compress_mode,
    matrix_XYZ_to_RGB,
) -> NDArrayFloat:

    luminanceXYZ = JMh_to_XYZ(
        JMh, XYZ_w, L_A, Y_b, surround, discount_illuminant, matrix_lms, compress_mode
    )
    luminanceRGB = vector_dot(matrix_XYZ_to_RGB, luminanceXYZ)
    return luminanceRGB


def luminance_RGB_to_JMh(
    luminanceRGB,
    XYZ_w: ArrayLike,
    L_A: ArrayLike,
    Y_b: ArrayLike,
    surround,
    discount_illuminant,
    matrix_lms,
    compress_mode,
    matrix_RGB_to_XYZ,
) -> NDArrayFloat:

    XYZ = vector_dot(matrix_RGB_to_XYZ, luminanceRGB)
    JMh = XYZ_to_JMh(
        XYZ, XYZ_w, L_A, Y_b, surround, discount_illuminant, matrix_lms, compress_mode
    )
    return JMh


# convert achromatic luminance to Hellwig J
def Y_to_J(
    Y,
    L_A,
    Y_b,
    surround,
):
    surround = VIEWING_CONDITIONS_HELLWIG2022[surround]

    k = 1.0 / (5.0 * L_A + 1.0)
    k4 = k * k * k * k
    F_L = 0.2 * k4 * (5.0 * L_A) + 0.1 * np.power((1.0 - k4), 2.0) * np.power(
        5.0 * L_A, 1.0 / 3.0
    )
    n = Y_b / 100.0
    z = 1.48 + np.sqrt(n)
    F_L_W = np.power(F_L, 0.42)
    A_w = (400.0 * F_L_W) / (27.13 + F_L_W)

    F_L_Y = np.power(F_L * abs(Y) / 100.0, 0.42)

    return np.sign(Y) * (
        100.0 * np.power(((400.0 * F_L_Y) / (27.13 + F_L_Y)) / A_w, surround.c * z)
    )
