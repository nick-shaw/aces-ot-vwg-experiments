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

from drt_cusp_lib import cuspFromTable

from drt_cam import JMh_to_luminance_RGB


# reimplemented from https:#github.com/nick-shaw/aces-ot-vwg-experiments/blob/master/python/intersection_approx.py
def findGamutBoundaryIntersection(
    JMh_s, JM_cusp, J_focus, J_max, slope_gain, smoothness, gamma_top, gamma_bottom, params
):
    JM_source = JMh_s[..., :2]

    slope = 0.0

    s = max(0.000001, smoothness)
    JM_smoothCusp = JM_cusp.copy()
    JM_smoothCusp[..., 0] *= 1.0 + params.smoothCuspJ * s  # J
    JM_smoothCusp[..., 1] *= 1.0 + params.smoothCuspM * s  # M

    J_intersect_source = solve_J_intersect(JM_source, J_focus, J_max, slope_gain)
    J_intersect_cusp = solve_J_intersect(JM_smoothCusp, J_focus, J_max, slope_gain)

    slope = np.where(
        J_intersect_source < J_focus,
        J_intersect_source * (J_intersect_source - J_focus) / (J_focus * slope_gain),
        (J_max - J_intersect_source)
        * (J_intersect_source - J_focus)
        / (J_focus * slope_gain),
    )

    M_boundary_lower = (
        J_intersect_cusp
        * np.power(J_intersect_source / J_intersect_cusp, 1 / gamma_bottom)
        / (JM_smoothCusp[..., 0] / JM_smoothCusp[..., 1] - slope)
    )

    M_boundary_upper = (
        JM_smoothCusp[..., 1]
        * (J_max - J_intersect_cusp)
        * np.power(
            (J_max - J_intersect_source) / (J_max - J_intersect_cusp), 1.0 / gamma_top
        )
        / (slope * JM_smoothCusp[..., 1] + J_max - JM_smoothCusp[..., 0])
    )

    M_boundary = JM_smoothCusp[..., 1] * smin(
        M_boundary_lower / JM_smoothCusp[..., 1], M_boundary_upper / JM_smoothCusp[..., 1], s
    )

    # J_boundary is not actually needed, but the calculation would be as follows
    J_boundary = J_intersect_source + slope * M_boundary

    return tstack([J_boundary, M_boundary, J_intersect_source])


# reimplemented from https://github.com/nick-shaw/aces-ot-vwg-experiments/blob/master/python/intersection_approx.py
def solve_J_intersect(JM, focusJ, maxJ, slope_gain):
    a = JM[..., 1] / (focusJ * slope_gain)
    b = 0.0
    c = 0.0
    intersectJ = 0.0

    b = np.where(
        JM[..., 0] < focusJ,
        1.0 - JM[..., 1] / slope_gain,
        -(1.0 + JM[..., 1] / slope_gain + maxJ * JM[..., 1] / (focusJ * slope_gain)),
    )

    c = np.where(
        JM[..., 0] < focusJ, -JM[..., 0], maxJ * JM[..., 1] / slope_gain + JM[..., 0]
    )

    root = np.sqrt(b * b - 4.0 * a * c)

    intersectJ = np.where(
        JM[..., 0] < focusJ, 2.0 * c / (-b - root), 2.0 * c / (-b + root)
    )

    return intersectJ


# Smooth minimum of a and b
def smin(a, b, s):
    h = np.maximum(s - np.abs(a - b), np.zeros(a.shape)) / s
    return np.minimum(a, b) - h * h * h * s * (1.0 / 6.0)


def lerp(a, b, t):
    return a + t * (b - a)


# https://www.desmos.com/calculator/oe2fscya80
def getFocusGain(
    J,
    cuspJ,
    params,
):

    thr = lerp(cuspJ, params.limitJmax, params.focusGainBlend)

    # Approximate inverse required above threshold
    jmin = np.minimum(np.full(J.shape, params.limitJmax), J)
    jbound = np.maximum(np.full(J.shape, 0.0001), params.limitJmax - jmin)
    gain = (params.limitJmax - thr) / jbound
    gain = np.power(np.log10(gain), 1.0 / params.focusAdjustGain) + 1.0

    return np.where(J > thr, gain, np.ones(J.shape))


def getCompressionFuncParams(
    JMh,
    params,
):

    locusMax = getReachBoundary(
        JMh, params.gamutCuspTableReach, params
    )[..., 1]
    difference = np.maximum(np.full(locusMax.shape, 1.0001), locusMax / JMh[..., 1])
    threshold = np.maximum(params.compressionFuncParams[0], 1.0 / difference)
    return tstack(
        [
            threshold,
            difference,
            difference,
            np.full(threshold.shape, params.compressionFuncParams[3]),
        ]
    )


def getReachBoundary(JMh, table, params):

    J, M, h = tsplit(JMh)

    reachMaxM = np.interp(h, table[..., 2], table[..., 1], period=360)

    JMcusp = cuspFromTable(h, params.gamutCuspTable)
    focusJ = lerp(
        JMcusp[..., 0],
        params.midJ,
        np.minimum(
            np.ones(JMcusp[..., 0].shape),
            params.cuspMidBlend - (JMcusp[..., 0] / params.limitJmax),
        ),
    )
    slope_gain = (
        params.limitJmax
        * params.focusDist
        * getFocusGain(J, JMcusp[..., 0], params)
    )
    intersectJ = solve_J_intersect(
        tstack([J, M]), focusJ, params.limitJmax, slope_gain
    )
    slope = np.where(
        intersectJ < focusJ,
        intersectJ * (intersectJ - focusJ) / (focusJ * slope_gain),
        (params.limitJmax - intersectJ)
        * (intersectJ - focusJ)
        / (focusJ * slope_gain),
    )

    boundaryNick = (
        params.limitJmax
        * np.power(intersectJ / params.limitJmax, params.model_gamma)
        * reachMaxM
        / (params.limitJmax - slope * reachMaxM)
    )
    return tstack([J, boundaryNick, h])


# "PowerP" compression function (also used in the ACES Reference Gamut Compression transform)
# values of v above  'treshold' are compressed by a 'power' function
# so that an input value of 'limit' results in an output of 1.0
def compressPowerP(v, threshold, limit, power, inverse):
    s = (limit - threshold) / np.power(
        np.power((1.0 - threshold) / (limit - threshold), -power) - 1.0, 1.0 / power
    )

    if inverse:
        vCompressed = np.where(
            np.logical_or(
                v < threshold, np.logical_or(limit < 1.0001, v > threshold + s)
            ),
            v,
            threshold
            + s
            * np.power(
                -(
                    np.power((v - threshold) / s, power)
                    / (np.power((v - threshold) / s, power) - 1.0)
                ),
                1.0 / power,
            ),
        )
    else:
        vCompressed = np.where(
            np.logical_or(v < threshold, limit < 1.0001),
            v,
            threshold
            + s
            * ((v - threshold) / s)
            / (np.power(1.0 + np.power((v - threshold) / s, power), 1.0 / power)),
        )

    return vCompressed


def evaluate_gamma_fit(JMcusp, testJmh, topGamma, boundaryRGB, params):
    testJmh = np.asarray(testJmh)
    testJ, testM, testh = tsplit(testJmh)
    Jcusp, Mcusp = tsplit(JMcusp)

    focusJ = lerp(
        Jcusp,
        params.midJ,
        np.minimum(1.0, params.cuspMidBlend - (Jcusp / params.limitJmax))
    )

    slope_gain  = params.limitJmax * params.focusDist * getFocusGain(testJ, Jcusp, params)
    approxLimit = findGamutBoundaryIntersection(testJmh, JMcusp, focusJ, params.limitJmax, slope_gain, params.smoothCusps, topGamma, params.gamutBottomGamma, params)
    approximate_JMh = tstack([
        approxLimit[..., 0], approxLimit[..., 1], testh])

    newLimitRGB = JMh_to_luminance_RGB(
        approximate_JMh,
        params.limit_whiteXYZ,
        params.L_A,
        params.Y_b,
        params.limit_viewingConditions,
        params.limit_discountIlluminant,
        params.matrix_lms,
        params.compress_mode,
        params.limit_XYZ_to_RGB,
    )
    newLimitRGB = newLimitRGB / boundaryRGB / params.referenceLuminance

    return np.all(np.any(newLimitRGB > 1, axis=-1))
