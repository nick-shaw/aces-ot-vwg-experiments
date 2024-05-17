from collections import namedtuple
from typing import Callable

from dataclasses import dataclass

import numpy as np

import colour
from colour.hints import ArrayLike, NDArrayFloat, Tuple
from colour.algebra import vector_dot
from colour.utilities import (
    as_float,
    as_float_array,
    from_range_100,
    from_range_degrees,
    ones,
    tsplit,
    tstack,
)

from drt_colour_lib import HSV_to_RGB
from drt_cusp_lib import cuspFromTable
from drt_gamut_compress_lib import evaluate_gamma_fit
from drt_cam import (
    VIEWING_CONDITIONS_HELLWIG2022,
    Y_to_Hellwig_J,
    luminance_RGB_to_JMh,
    JMh_to_luminance_RGB,
)

@dataclass
class DRTParams:
    # CAM
    referenceLuminance: float
    L_A: float
    Y_b: float
    matrix_lms: ArrayLike
    model_gamma: float
    ap1_clamp: bool

    # Tonescale
    peakLuminance: float
    n: float
    n_r: float
    g: float
    t_1: float
    c_t: float
    s_2: float
    u_2: float
    m_2: float

    # Chroma Compression
    chroma_compress: float
    chroma_compress_fact: float
    chroma_expand: float
    chroma_expand_fact: float
    chroma_expand_thr: float
    limitJmax: float
    sat: float
    sat_thr: float
    compr: float
    applyReachClamp: bool
    ccreach_RGB_to_XYZ: ArrayLike
    ccreach_XYZ_to_RGB: ArrayLike

    # Gamut Compression:
    cuspMidBlend: float
    focusDistance: float
    focusDist: float
    focusAdjustGain: float
    focusGainBlend: float
    focusDistScaling: float
    midJ: float
    smoothCusps: float
    smoothCuspM: float
    compressionFuncParams: ArrayLike
    # Not used, assuming same as cc (AP1)
    # gcreach_RGB_to_XYZ: ArrayLike
    # gcreach_XYZ_to_RGB: ArrayLike

    # Cusp / hull tables
    tableSize: int
    gamutCuspTable: ArrayLike
    gamutCuspTableReach: ArrayLike
    cgamutCuspTable: ArrayLike
    cgamutReachTable: ArrayLike
    gamutTopGamma: ArrayLike
    gamutBottomGamma: float

    # Input
    input_discountIlluminant: bool
    input_viewingConditions: str
    input_XYZ_to_RGB: ArrayLike
    input_RGB_to_XYZ: ArrayLike
    input_whiteXYZ: ArrayLike

    # Limit
    limit_discountIlluminant: bool
    limit_viewingConditions: str
    limit_XYZ_to_RGB: ArrayLike
    limit_RGB_to_XYZ: ArrayLike
    limit_whiteXYZ: ArrayLike

    # Output
    output_discountIlluminant: bool
    output_viewingConditions: str
    output_XYZ_to_RGB: ArrayLike
    output_RGB_to_XYZ: ArrayLike
    eotf: Callable
    eotf_inverse: Callable
    fitWhite: bool
    softclamp: bool
    clamp_thr: float
    clamp_dist: float
    clamp: bool


def drt_params(target):
    if target == 1:
        sRGBDerived = colour.models.RGB_COLOURSPACE_sRGB
        sRGBDerived.use_derived_transformation_matrices(True)
        return _drt_params(
            inputDiscountIlluminant=True,
            inputViewingConditions="Dim",
            inputColourSpace=colour.models.RGB_COLOURSPACE_ACES2065_1,
            peakLuminance=100,
            limitDiscountIlluminant=True,
            limitViewingConditions="Dim",
            limitColourSpace=sRGBDerived,
            outputDiscountIlluminant=True,
            outputViewingConditions="Dim",
            outputColourSpace=sRGBDerived,
        )
    elif target == 2:

        def g24_encode(x):
            return np.power(x, 1 / 2.4, where=x > 0)

        def g24_decode(x):
            return np.power(x, 2.4, where=x > 0)

        BT709_CS = colour.models.RGB_COLOURSPACE_BT709
        BT1886_709_CS = colour.models.RGB_Colourspace(
            "Rec709 Gamma 2.4",
            BT709_CS.primaries,
            BT709_CS.whitepoint,
            "D65",
            BT709_CS.matrix_RGB_to_XYZ,
            BT709_CS.matrix_XYZ_to_RGB,
            cctf_encoding=g24_encode,
            cctf_decoding=g24_decode,
        )

        return _drt_params(
            inputDiscountIlluminant=True,
            inputViewingConditions="Dim",
            inputColourSpace=colour.models.RGB_COLOURSPACE_ACES2065_1,
            peakLuminance=100,
            limitDiscountIlluminant=True,
            limitViewingConditions="Dim",
            limitColourSpace=BT1886_709_CS,
            outputDiscountIlluminant=True,
            outputViewingConditions="Dim",
            outputColourSpace=sRGBDerived,
        )
    elif target == 3:

        def pq_encode(x):
            return colour.models.eotf_inverse_BT2100_PQ(x * 100)

        def pq_decode(x):
            return colour.models.eotf_BT2100_PQ(x) / 100

        BT2020_CS = colour.models.RGB_COLOURSPACE_BT2020
        BT2100_PQ_CS = colour.models.RGB_Colourspace(
            "Rec2100 PQ",
            BT2020_CS.primaries,
            BT2020_CS.whitepoint,
            "D65",
            BT2020_CS.matrix_RGB_to_XYZ,
            BT2020_CS.matrix_XYZ_to_RGB,
            cctf_encoding=pq_encode,
            cctf_decoding=pq_decode,
        )

        return _drt_params(
            inputDiscountIlluminant=True,
            inputViewingConditions="Dim",
            inputColourSpace=colour.models.RGB_COLOURSPACE_ACES2065_1,
            peakLuminance=1000,
            limitDiscountIlluminant=True,
            limitViewingConditions="Dim",
            limitColourSpace=colour.models.RGB_COLOURSPACE_DISPLAY_P3,
            outputDiscountIlluminant=True,
            outputViewingConditions="Dim",
            outputColourSpace=BT2100_PQ_CS,
        )
    else:
        raise RuntimeError(f"Unsupported target {target}")


def _drt_params(
    inputDiscountIlluminant=True,
    inputViewingConditions="Dim",
    inputColourSpace=colour.models.RGB_COLOURSPACE_ACES2065_1,
    peakLuminance=100,
    limitDiscountIlluminant=True,
    limitViewingConditions="Dim",
    limitColourSpace=colour.models.RGB_COLOURSPACE_sRGB,
    outputDiscountIlluminant=True,
    outputViewingConditions="Dim",
    outputColourSpace=colour.models.RGB_COLOURSPACE_sRGB,
):
    # LMS cone space for Hellwig 2022
    rxy = np.array([0.8336, 0.1735])
    gxy = np.array([2.3854, -1.4659])
    bxy = np.array([0.087, -0.125])
    wxy = np.array([0.333, 0.333])
    CUSTOM_CAT16 = RGBPrimsToXYZMatrix(rxy, gxy, bxy, wxy, 1, 1)

    # Tone scale
    peakLuminanceTS = peakLuminance
    de_params = daniele_evo_params(peakLuminanceTS)

    # White
    white = np.array([1.0, 1.0, 1.0])
    inputWhiteXYZ = vector_dot(inputColourSpace.matrix_RGB_to_XYZ, white) * 100
    limitWhiteXYZ = vector_dot(limitColourSpace.matrix_RGB_to_XYZ, white) * 100

    params = DRTParams(
        # CAM
        referenceLuminance=100,
        L_A=100,
        Y_b=20,
        matrix_lms=CUSTOM_CAT16,
        model_gamma=None,
        ap1_clamp=True,

        # Tonescale
        peakLuminance=peakLuminance,
        n=de_params["n"],
        n_r=de_params["n_r"],
        g=de_params["g"],
        t_1=de_params["t_1"],
        c_t=de_params["c_t"],
        s_2=de_params["s_2"],
        u_2=de_params["u_2"],
        m_2=de_params["m_2"],

        # Chroma Compression
        chroma_compress=2.4,
        chroma_compress_fact=3.3,
        chroma_expand=1.3,
        chroma_expand_fact=0.69,
        chroma_expand_thr=0.5,
        limitJmax=None,
        sat=None,
        sat_thr=None,
        compr=None,
        applyReachClamp=None,
        ccreach_RGB_to_XYZ=colour.models.RGB_COLOURSPACE_ACESCG.matrix_RGB_to_XYZ,
        ccreach_XYZ_to_RGB=colour.models.RGB_COLOURSPACE_ACESCG.matrix_XYZ_to_RGB,

        # Gamut Compression
        cuspMidBlend=1.3,
        focusDistance=1.35,
        focusDist=None,
        focusAdjustGain=0.55,
        focusGainBlend=0.3,
        focusDistScaling=1.75,
        midJ=None,
        smoothCusps=0.12,
        smoothCuspM=0.27,
        compressionFuncParams=[0.75, 1.1, 1.3, 1],
        # gcreach_RGB_to_XYZ=colour.models.RGB_COLOURSPACE_ACESCG.matrix_RGB_to_XYZ,
        # gcreach_XYZ_to_RGB=colour.models.RGB_COLOURSPACE_ACESCG.matrix_XYZ_to_RGB,

        # Cusp / hull tables
        tableSize=360,
        gamutCuspTable=None,
        gamutCuspTableReach=None,
        cgamutCuspTable=None,
        cgamutReachTable=None,
        gamutTopGamma=None,
        gamutBottomGamma=1/1.14,

        # Input
        input_discountIlluminant=inputDiscountIlluminant,
        input_viewingConditions=inputViewingConditions,
        input_XYZ_to_RGB=inputColourSpace.matrix_XYZ_to_RGB,
        input_RGB_to_XYZ=inputColourSpace.matrix_RGB_to_XYZ,
        input_whiteXYZ=inputWhiteXYZ,

        # Limit
        limit_discountIlluminant=limitDiscountIlluminant,
        limit_viewingConditions=limitViewingConditions,
        limit_XYZ_to_RGB=limitColourSpace.matrix_XYZ_to_RGB,
        limit_RGB_to_XYZ=limitColourSpace.matrix_RGB_to_XYZ,
        limit_whiteXYZ=limitWhiteXYZ,

        # Output
        output_discountIlluminant=outputDiscountIlluminant,
        output_viewingConditions=outputViewingConditions,
        output_XYZ_to_RGB=outputColourSpace.matrix_XYZ_to_RGB,
        output_RGB_to_XYZ=outputColourSpace.matrix_RGB_to_XYZ,
        eotf=outputColourSpace.cctf_decoding,
        eotf_inverse=outputColourSpace.cctf_encoding,
        fitWhite=False,
        softclamp=True,
        clamp_thr=0.99,
        clamp_dist=1.1,
        clamp=True,
    )

    # Derived parameters

    # Chroma Compression
    limitJmax = Y_to_Hellwig_J(
        peakLuminanceTS, params.L_A, params.Y_b, inputViewingConditions
    )

    surround = VIEWING_CONDITIONS_HELLWIG2022[inputViewingConditions]
    model_gamma = 1.0 / (surround.c * (1.48 + np.sqrt(params.Y_b / params.L_A)))
    log_peak = np.log10(params.n / params.n_r)
    compr = params.chroma_compress + (params.chroma_compress * params.chroma_compress_fact) * log_peak
    sat = max(
        0.2, params.chroma_expand - (params.chroma_expand * params.chroma_expand_fact) * log_peak
    )
    sat_thr = params.chroma_expand_thr / params.n

    params.limitJmax = limitJmax
    params.model_gamma = model_gamma
    params.sat = sat
    params.sat_thr = sat_thr
    params.compr = compr
    params.applyReachClamp = False

    # Gamut Compression
    midJ = Y_to_Hellwig_J(
        params.c_t * params.n_r,
        params.L_A,
        params.Y_b,
        inputViewingConditions,
    )

    # Gamut mapper focus distance scaling with peak luminance for
    # HDR/SDR appearance match.  The projection gets slightly less
    # steep with higher peak luminance.
    # https:#www.desmos.com/calculator/bnfhjcq5vf
    focusDist = params.focusDistance + params.focusDistance * params.focusDistScaling * log_peak

    params.focusDist = focusDist
    params.midJ = midJ

    # Cusp tables

    cusp_tables(params)

    return params


def RGBPrimsToXYZMatrix(rxy, gxy, bxy, wxy, Y, direction):
    # given r g b chromaticities and whitepoint, convert RGB colors to XYZ
    # based on CtlColorSpace.cpp from the CTL source code : 77
    # param: xy - dict of chromaticity xy coordinates: rxy: float2(x, y) etc
    # param: Y - luminance of "white" - defaults to 1.0
    # param: inverse - calculate XYZ to RGB instead

    r = rxy
    g = gxy
    b = bxy
    w = wxy

    X = w[0] * Y / w[1]
    Z = (1 - w[0] - w[1]) * Y / w[1]

    # Scale factors for matrix rows
    d = r[0] * (b[1] - g[1]) + b[0] * (g[1] - r[1]) + g[0] * (r[1] - b[1])

    Sr = (
        X * (b[1] - g[1])
        - g[0] * (Y * (b[1] - 1.0) + b[1] * (X + Z))
        + b[0] * (Y * (g[1] - 1.0) + g[1] * (X + Z))
    ) / d

    Sg = (
        X * (r[1] - b[1])
        + r[0] * (Y * (b[1] - 1.0) + b[1] * (X + Z))
        - b[0] * (Y * (r[1] - 1.0) + r[1] * (X + Z))
    ) / d

    Sb = (
        X * (g[1] - r[1])
        - r[0] * (Y * (g[1] - 1.0) + g[1] * (X + Z))
        + g[0] * (Y * (r[1] - 1.0) + r[1] * (X + Z))
    ) / d

    # Assemble the matrix
    Mdata = np.array(
        [
            [Sr * r[0], Sr * r[1], Sr * (1.0 - r[0] - r[1])],
            [Sg * g[0], Sg * g[1], Sg * (1.0 - g[0] - g[1])],
            [Sb * b[0], Sb * b[1], Sb * (1.0 - b[0] - b[1])],
        ]
    )

    MdataNukeOrder = Mdata.T

    newMatrix = MdataNukeOrder

    # create inverse matrix
    newMatrixInverse = np.linalg.inv(newMatrix)

    # return forward or inverse matrix
    if direction == 0:
        return newMatrix
    elif direction == 1:
        return newMatrixInverse


def daniele_evo_params(peakLuminance):

    # DanieleEvoCurve (ACES2 candidate) parameters
    daniele_n = peakLuminance  # peak white
    daniele_n_r = 100.0  # Normalized white in nits (what 1.0 should be)
    daniele_g = 1.15  # surround / contrast
    daniele_c = 0.18  # scene-referred grey
    daniele_c_d = 10.013  # display-referred grey (in nits)
    daniele_w_g = 0.14  # grey change between different peak luminance
    daniele_t_1 = (
        0.04  # shadow toe, flare/glare compensation - how ever you want to call it
    )
    daniele_r_hit_min = 128.0  # Scene-referred value "hitting the roof" at 100 nits
    daniele_r_hit_max = 896.0  # Scene-referred value "hitting the roof" at 10,000 nits

    # pre-calculate Daniele Evo constants
    daniele_r_hit = daniele_r_hit_min + (daniele_r_hit_max - daniele_r_hit_min) * (
        np.log(daniele_n / daniele_n_r) / np.log(10000.0 / 100.0)
    )
    daniele_m_0 = daniele_n / daniele_n_r
    daniele_m_1 = 0.5 * (
        daniele_m_0 + np.sqrt(daniele_m_0 * (daniele_m_0 + 4.0 * daniele_t_1))
    )
    daniele_u = pow(
        (daniele_r_hit / daniele_m_1) / ((daniele_r_hit / daniele_m_1) + 1.0), daniele_g
    )
    daniele_m = daniele_m_1 / daniele_u
    daniele_w_i = np.log(daniele_n / 100.0) / np.log(2.0)
    daniele_c_t = daniele_c_d * (1.0 + daniele_w_i * daniele_w_g) / daniele_n_r
    daniele_g_ip = 0.5 * (
        daniele_c_t + np.sqrt(daniele_c_t * (daniele_c_t + 4.0 * daniele_t_1))
    )
    daniele_g_ipp2 = (
        -daniele_m_1
        * pow(daniele_g_ip / daniele_m, 1.0 / daniele_g)
        / (pow(daniele_g_ip / daniele_m, 1.0 / daniele_g) - 1.0)
    )
    daniele_w_2 = daniele_c / daniele_g_ipp2
    daniele_s_2 = daniele_w_2 * daniele_m_1
    daniele_u_2 = pow(
        (daniele_r_hit / daniele_m_1) / ((daniele_r_hit / daniele_m_1) + daniele_w_2),
        daniele_g,
    )
    daniele_m_2 = daniele_m_1 / daniele_u_2

    return {
        "n"   : daniele_n,
        "n_r" : daniele_n_r,
        "g"   : daniele_g,
        "t_1" : daniele_t_1,
        "c_t" : daniele_c_t,
        "s_2" : daniele_s_2,
        "u_2" : daniele_u_2,
        "m_2" : daniele_m_2,
    }


def cusp_tables(params):

    h_samples = params.tableSize

    # Cusp table for chroma compression gamut

    gamutCuspTableUnsorted = np.zeros((h_samples, 3))
    h = np.linspace(0, 1, h_samples, endpoint=False)
    ones = np.ones(h.shape)
    RGB = HSV_to_RGB(tstack([h, ones, ones]))
    RGB *= params.peakLuminance
    gamutCuspTableUnsorted = luminance_RGB_to_JMh(
        RGB,
        params.input_whiteXYZ,
        params.L_A,
        params.Y_b,
        params.input_viewingConditions,
        params.input_discountIlluminant,
        params.matrix_lms,
        params.ccreach_RGB_to_XYZ,
    )
    minhIndex = np.argmin(gamutCuspTableUnsorted[..., 2], axis=-1)
    cgamutCuspTable = np.roll(gamutCuspTableUnsorted, -minhIndex, axis=0)

    # Reach table for the chroma compression reach.

    cgamutReachTable = np.zeros((h_samples, 3))
    for i in range(h_samples):
        hue = float(i) * 360 / h_samples
        cgamutReachTable[i][2] = hue

        # Initially tried binary search between 0 and 1300, but there must be cases where extreme values wrap
        # So start small and jump in small ish steps until we are outside then binary search inside that range
        search_range = 100.0
        low, high = 0.0, search_range
        outside = False
        while not outside and high < 1400:
            JMhSearch = np.array([params.limitJmax, high, hue])
            newLimitRGB = JMh_to_luminance_RGB(
                JMhSearch,
                params.input_whiteXYZ,
                params.L_A,
                params.Y_b,
                params.limit_viewingConditions,
                params.limit_discountIlluminant,
                params.matrix_lms,
                params.ccreach_XYZ_to_RGB,
            )
            newLimitRGB /= params.peakLuminance
            outside = np.any(newLimitRGB < 0, axis=-1)
            if not outside:
                low = high
                high = high + search_range

        while (high - low) > 1e-4: # how close should we be
            sampleM = (high + low) / 2
            JMhSearch = np.array([params.limitJmax, sampleM, hue])
            newLimitRGB = JMh_to_luminance_RGB(
                JMhSearch,
                params.input_whiteXYZ,
                params.L_A,
                params.Y_b,
                params.input_viewingConditions,
                params.input_discountIlluminant,
                params.matrix_lms,
                params.ccreach_XYZ_to_RGB,
            )
            newLimitRGB /= params.peakLuminance
            outside = np.any(newLimitRGB < 0, axis=-1)
            if outside:
                high = sampleM
            else:
                low = sampleM

        cgamutReachTable[i][0] = params.limitJmax
        cgamutReachTable[i][1] = high

    # Cusp table for limiting gamut

    gamutCuspTableUnsorted = np.zeros((h_samples, 3))
    h = np.linspace(0, 1, h_samples, endpoint=False)
    ones = np.ones(h.shape)
    RGB = HSV_to_RGB(tstack([h, ones, ones]))
    RGB *= params.peakLuminance
    gamutCuspTableUnsorted = luminance_RGB_to_JMh(
        RGB,
        params.limit_whiteXYZ,
        params.L_A,
        params.Y_b,
        params.input_viewingConditions,
        params.input_discountIlluminant,
        params.matrix_lms,
        params.limit_RGB_to_XYZ,
    )
    minhIndex = np.argmin(gamutCuspTableUnsorted[..., 2], axis=-1)
    gamutCuspTable = np.roll(gamutCuspTableUnsorted, -minhIndex, axis=0)

    # Cusp table for gamut compressor limiting reach gamut.

    gamutCuspTableReach = cgamutReachTable

    # Upper hull gamma values for the gamut mapper

    testPositions = [0.01, 0.5, 0.99]
    gamutTopGamma = np.full(h_samples, -1.0)
    for i in range(h_samples):
        hue = float(i) * 360 / h_samples
        JMcusp = cuspFromTable(hue, gamutCuspTable)
        testJmh = [
            np.array([
                JMcusp[0] + ((params.limitJmax - JMcusp[0]) * testPosition),
                JMcusp[1],
                hue
            ])
            for testPosition in testPositions
        ]

        search_range = 0.4
        low = 0.4
        high = low + search_range
        gamma_found = False

        while not gamma_found and high < 5.0:
            gamma_found = evaluate_gamma_fit(JMcusp, testJmh, high, params)
            if not gamma_found:
                low = high
                high = high + search_range

        testGamma = -1.0
        while (high - low) > 1e-5:
            testGamma = (high + low) / 2
            gamma_found = evaluate_gamma_fit(JMcusp, testJmh, testGamma, params)
            if gamma_found:
                high = testGamma
            else:
                low = testGamma

        gamutTopGamma[i] = testGamma

    params.cgamutCuspTable = cgamutCuspTable
    params.cgamutReachTable = cgamutReachTable
    params.gamutCuspTable = gamutCuspTable
    params.gamutCuspTableReach = gamutCuspTableReach
    params.gamutTopGamma = gamutTopGamma
