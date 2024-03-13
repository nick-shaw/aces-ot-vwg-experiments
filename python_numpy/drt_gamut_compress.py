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

from drt_cusp_lib import (
    cuspFromTable,
    hueDependantUpperHullGamma,
)
from drt_gamut_compress_lib import (
    findGamutBoundaryIntersection,
    getFocusGain,
    getCompressionFuncParams,
    compressPowerP,
)


def gamutCompressForward(JMh, params):
    return compressGamut2(JMh, 0, JMh[..., 0], params)


def gamutCompressInverse(JMh, params):
    J, M, h = tsplit(JMh)
    JMcusp = cuspFromTable(h, params.gamutCuspTable)
    Jcusp, Mcusp = tsplit(JMcusp)

    JMhcompress = compressGamut2(JMh, 1, J, params)

    cond = J <= lerp(Jcusp, params.limitJmax, params.focusGainBlend)
    cond = np.expand_dims(cond, axis=-1)

    return np.where(
        cond,
        # Analytic inverse below threshold
        JMhcompress,
        # Approximation
        compressGamut2(JMh, 1, JMhcompress[..., 0], params),
    )


def compressGamut2(
    JMh,
    invert,
    Jx,
    params,
):
    J, M, h = tsplit(JMh)
    project_from = tstack([J, M])
    JMcusp = cuspFromTable(h, params.gamutCuspTable)
    Jcusp, Mcusp = tsplit(JMcusp)

    # Calculate where the out of gamut color is projected to
    t = np.minimum(
        np.ones(Jcusp.shape), params.cuspMidBlend - (Jcusp / params.limitJmax)
    )
    focusJ = lerp(Jcusp, params.midJ, t)

    # https://www.desmos.com/calculator/9u0wiiz9ys
    Mratio = M / (params.focusDist * Mcusp)
    a = np.maximum(np.full(focusJ.shape, 0.001), Mratio / focusJ)
    b0 = 1.0 - Mratio
    b1 = -(1.0 + Mratio + (a * params.limitJmax))
    b = np.where(J < focusJ, b0, b1)
    c0 = -J
    c1 = J + params.limitJmax * Mratio
    c = np.where(J < focusJ, c0, c1)

    # XXX this sqrt can cause NaNs (subtraction goes negative)
    J0 = np.sqrt(b * b - 4 * a * c)
    J1 = (-b - J0) / (2 * a)
    J0 = (-b + J0) / (2 * a)
    projectJ = np.where(J < focusJ, J0, J1)

    slope_gain = (
        params.limitJmax * params.focusDist * getFocusGain(J, Jcusp, params)
    )

    # Find gamut intersection
    gamma_top = hueDependantUpperHullGamma(h, params.gamutTopGamma)
    gamma_bottom = params.gamutBottomGamma

    nickBoundaryReturn = findGamutBoundaryIntersection(
        JMh,
        JMcusp,
        focusJ,
        params.limitJmax,
        slope_gain,
        params.smoothCusps,
        gamma_top,
        gamma_bottom,
        params,
    )
    JMboundary = nickBoundaryReturn[..., :2]
    project_to = tstack(
        [nickBoundaryReturn[..., 2], np.zeros(nickBoundaryReturn[..., 2].shape)]
    )
    projectJ = nickBoundaryReturn[..., 2]

    # Get hue dependent compression parameters
    comprParams = getCompressionFuncParams(
        tstack([JMboundary[..., 0], JMboundary[..., 1], h]),
        params,
    )

    # Compress the out of gamut color along the projection line
    JMcompressed = project_from

    lowerMlimit = 0.0001  # Testing a small value here

    v = M / JMboundary[..., 1]
    v = compressPowerP(
        v,
        comprParams[..., 0],
        lerp(comprParams[..., 2], comprParams[..., 1], projectJ / params.limitJmax),
        comprParams[..., 3],
        invert,
    )
    v = np.expand_dims(v, axis=-1)
    v = project_to + v * (JMboundary - project_to)

    # using a small value to test against here rather than 0.0, and I was getting Nans on inversion.
    cond = np.logical_and(J < params.limitJmax, M > lowerMlimit)
    cond = np.expand_dims(cond, axis=-1)

    JMcompressed = np.where(cond, v, tstack([J, np.zeros(J.shape)]))

    cond = M == 0
    cond = np.expand_dims(cond, axis=-1)

    return np.where(cond, JMh, tstack([JMcompressed[..., 0], JMcompressed[..., 1], h]))


def lerp(a, b, t):
    return a + t * (b - a)
