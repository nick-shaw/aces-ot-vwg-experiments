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


# Return compression gamut cusp M scaled with an eccentricity factor
def cusp_with_eccentricity_factor(
    h,
    cc_params,
    cusp_params,
):
    JMcusp = cuspFromTable(h, cusp_params.cgamutCuspTable)
    J, M = tsplit(JMcusp)

    e_t = 1.0

    return M * e_t


def cuspFromTable(h, table):

    h = np.asarray(h)

    res = np.zeros((*h.shape, 2))
    res[..., 0] = np.interp(h, table[..., 2], table[..., 0], period=360)
    res[..., 1] = np.interp(h, table[..., 2], table[..., 1], period=360)
    return res


def cReachFromTable(h, table):

    res = np.interp(h, table[..., 2], table[..., 1], period=360)
    return res


def hueDependantUpperHullGamma(h, table):

    xp = np.arange(len(table))
    res = np.interp(h, xp, table, period=360)
    return res
