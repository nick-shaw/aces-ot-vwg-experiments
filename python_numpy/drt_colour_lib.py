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


# Code in this module has been lifted and adapted from Colour Science library

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"


def RGB_to_HSV(RGB: ArrayLike) -> NDArrayFloat:

    maximum = np.amax(RGB, -1)
    delta = np.ptp(RGB, -1)

    V = maximum

    R, G, B = tsplit(RGB)

    with sdiv_mode():
        S = sdiv(delta, maximum)

        delta_R = sdiv(((maximum - R) / 6) + (delta / 2), delta)
        delta_G = sdiv(((maximum - G) / 6) + (delta / 2), delta)
        delta_B = sdiv(((maximum - B) / 6) + (delta / 2), delta)

    H = delta_B - delta_G
    H = np.where(maximum == G, (1 / 3) + delta_R - delta_B, H)
    H = np.where(maximum == B, (2 / 3) + delta_G - delta_R, H)
    H[np.asarray(H < 0)] += 1
    H[np.asarray(H > 1)] -= 1
    H[np.asarray(delta == 0)] = 0

    HSV = tstack([H, S, V])

    return HSV


def HSV_to_RGB(HSV: ArrayLike) -> NDArrayFloat:

    H, S, V = tsplit(HSV)

    h = as_float_array(H * 6)
    h[np.asarray(h == 6)] = 0

    i = np.floor(h)
    j = V * (1 - S)
    k = V * (1 - S * (h - i))
    l = V * (1 - S * (1 - (h - i)))  # noqa: E741

    i = tstack([i, i, i]).astype(np.uint8)

    RGB = np.choose(
        i,
        [
            tstack([V, l, j]),
            tstack([k, V, j]),
            tstack([j, V, l]),
            tstack([j, k, V]),
            tstack([l, j, V]),
            tstack([V, j, k]),
        ],
        mode="clip",
    )

    return RGB
