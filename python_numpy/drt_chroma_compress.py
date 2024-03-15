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

from drt_cusp_lib import cuspFromTable, cReachFromTable


def chromaCompressionForward(
    JMh,
    origJ,
    params
):
    J, M, h = tsplit(JMh)
    M_orig = M.copy()

    nJ = J / params.limitJmax
    snJ = np.maximum(1.0 - nJ, np.zeros(nJ.shape))
    Mnorm = cuspFromTable(h, params.cgamutCuspTable)[..., 1]
    limit = (
        np.power(nJ, params.model_gamma)
        * cReachFromTable(h, params.cgamutReachTable)
        / Mnorm
    )

    #
    # Rescaling of M with the tonescaled J to get the M to the same range as
    # J after the tonescale.  The rescaling uses the Hellwig2022 model gamma to
    # keep the M/J ratio correct (keeping the chromaticities constant).
    #
    M *= np.power(J / origJ, params.model_gamma)

    # Normalize M with the rendering space cusp M
    M /= Mnorm

    #
    # Expand the colorfulness by running the toe function in reverse.  The goal is to
    # expand less saturated colors less and more saturated colors more.  The expansion
    # increases saturation in the shadows and mid-tones but not in the highlights.
    # The 0.001 offset starts the expansions slightly above zero.  The sat_thr makes
    # the toe less aggressive near black to reduce the expansion of noise.
    #
    M = limit - toe_forward(
        limit - M,
        limit - 0.001,
        snJ * params.sat,
        np.sqrt(nJ * nJ + params.sat_thr),
    )

    #
    # Compress the colorfulness.  The goal is to compress less saturated colors more and
    # more saturated colors less, especially in the highlights.  This step creates the
    # saturation roll-off in the highlights, but attemps to preserve pure colors.  This
    # mostly affects highlights and mid-tones, and does not compress shadows.
    #
    M = toe_forward(M, limit, nJ * params.compr, snJ)

    # Clamp M to the rendering space
    if params.applyReachClamp:
        M = np.minimum(np.ones(M.shape) * limit, M)

    # Denormalize
    M *= Mnorm

    return np.where(M_orig == 0.0, np.zeros(M.shape), M)


def chromaCompressionInverse(
    JMh,
    origJ,
    params,
):
    J, M, h = tsplit(JMh)
    M_orig = M.copy()

    nJ = J / params.limitJmax
    snJ = np.maximum(1.0 - nJ, np.zeros(nJ.shape))
    Mnorm = cuspFromTable(h, params.cgamutCuspTable)[..., 1]
    limit = (
        np.power(nJ, params.model_gamma)
        * cReachFromTable(h, params.cgamutReachTable)
        / Mnorm
    )

    M /= Mnorm
    M = toe_inverse(M, limit, nJ * params.compr, snJ)
    M = limit - toe_inverse(
        limit - M,
        limit - 0.001,
        snJ * params.sat,
        np.sqrt(nJ * nJ + params.sat_thr),
    )
    M *= Mnorm

    M *= pow(J / origJ, -params.model_gamma)

    return np.where(M_orig == 0.0, np.zeros(M.shape), M)


# A "toe" function that remaps the given value x between 0 and limit.
# The k1 and k2 parameters change the size and shape of the toe.
# https://www.desmos.com/calculator/6vplvw14ti
def toe_forward(x, limit, k1, k2):

    k2 = np.maximum(k2, np.full(k2.shape, 0.001))
    k1 = np.sqrt(k1 * k1 + k2 * k2)
    k3 = (limit + k1) / (limit + k2)
    xt = 0.5 * (k3 * x - k1 + np.sqrt((k3 * x - k1) * (k3 * x - k1) + 4 * k2 * k3 * x))

    return np.where(x > limit, x, xt)


def toe_inverse(x, limit, k1, k2):

    k2 = np.maximum(k2, np.full(k2.shape, 0.001))
    k1 = np.sqrt(k1 * k1 + k2 * k2)
    k3 = (limit + k1) / (limit + k2)
    xt = (x * x + k1 * x) / (k3 * (x + k2))

    return np.where(x > limit, x, xt)
