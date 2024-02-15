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


def gamutCompressForward(
    JMh,
    cusp_params,
    gc_params,
):
    return compressGamut2(JMh, 0, JMh[..., 0], cusp_params, gc_params)


def gamutCompressInverse(
    JMh,
    cusp_params,
    gc_params,
):
    J, M, h = tsplit(JMh)
    JMcusp = cuspFromTable(h, cusp_params.gamutCuspTable)
    Jcusp, Mcusp = tsplit(JMcusp)

    JMhcompress = compressGamut2(JMh, 1, J, cusp_params, gc_params)

    cond = J <= lerp(Jcusp, gc_params.limitJmax, gc_params.focusGainBlend)
    cond = np.expand_dims(cond, axis=-1)

    return np.where(
        cond,
        # Analytic inverse below threshold
        JMhcompress,
        # Approximation
        compressGamut2(JMh, 1, JMhcompress[..., 0], cusp_params, gc_params),
    )


def compressGamut2(
    JMh,
    invert,
    Jx,
    cusp_params,
    gc_params,
):
    J, M, h = tsplit(JMh)
    project_from = tstack([J, M])
    JMcusp = cuspFromTable(h, cusp_params.gamutCuspTable)
    Jcusp, Mcusp = tsplit(JMcusp)

    # Calculate where the out of gamut color is projected to
    t = np.minimum(
        np.ones(Jcusp.shape), gc_params.cuspMidBlend - (Jcusp / gc_params.limitJmax)
    )
    focusJ = lerp(Jcusp, gc_params.midJ, t)

    # https://www.desmos.com/calculator/9u0wiiz9ys
    Mratio = M / (gc_params.focusDist * Mcusp)
    a = np.maximum(np.full(focusJ.shape, 0.001), Mratio / focusJ)
    b0 = 1.0 - Mratio
    b1 = -(1.0 + Mratio + (a * gc_params.limitJmax))
    b = np.where(J < focusJ, b0, b1)
    c0 = -J
    c1 = J + gc_params.limitJmax * Mratio
    c = np.where(J < focusJ, c0, c1)

    # XXX this sqrt can cause NaNs (subtraction goes negative)
    J0 = np.sqrt(b * b - 4 * a * c)
    J1 = (-b - J0) / (2 * a)
    J0 = (-b + J0) / (2 * a)
    projectJ = np.where(J < focusJ, J0, J1)

    slope_gain = (
        gc_params.limitJmax * gc_params.focusDist * getFocusGain(J, Jcusp, gc_params)
    )

    # Find gamut intersection
    gamma_top = hueDependantUpperHullGamma(h, cusp_params.gamutTopGamma)
    gamma_bottom = cusp_params.gamutBottomGamma

    nickBoundaryReturn = findGamutBoundaryIntersection(
        JMh,
        JMcusp,
        focusJ,
        gc_params.limitJmax,
        slope_gain,
        gc_params.smoothCusps,
        gamma_top,
        gamma_bottom,
    )
    JMboundary = nickBoundaryReturn[..., :2]
    project_to = tstack(
        [nickBoundaryReturn[..., 2], np.zeros(nickBoundaryReturn[..., 2].shape)]
    )
    projectJ = nickBoundaryReturn[..., 2]

    # Get hue dependent compression parameters
    comprParams = getCompressionFuncParams(
        tstack([JMboundary[..., 0], JMboundary[..., 1], h]),
        gc_params,
        cusp_params,
    )

    # Compress the out of gamut color along the projection line
    JMcompressed = project_from

    lowerMlimit = 0.0001  # Testing a small value here

    v = M / JMboundary[..., 1]
    v = compressPowerP(
        v,
        comprParams[..., 0],
        lerp(comprParams[..., 2], comprParams[..., 1], projectJ / gc_params.limitJmax),
        comprParams[..., 3],
        invert,
    )
    v = np.expand_dims(v, axis=-1)
    v = project_to + v * (JMboundary - project_to)

    # using a small value to test against here rather than 0.0, and I was getting Nans on inversion.
    cond = np.logical_and(J < gc_params.limitJmax, M > lowerMlimit)
    cond = np.expand_dims(cond, axis=-1)

    JMcompressed = np.where(cond, v, tstack([J, np.zeros(J.shape)]))

    cond = M == 0
    cond = np.expand_dims(cond, axis=-1)

    return np.where(cond, JMh, tstack([JMcompressed[..., 0], JMcompressed[..., 1], h]))


def lerp(a, b, t):
    return a + t * (b - a)
