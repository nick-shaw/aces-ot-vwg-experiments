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
from drt_cam import (
    VIEWING_CONDITIONS_HELLWIG2022,
    Y_to_J,
    luminance_RGB_to_JMh,
    JMh_to_luminance_RGB,
)

# TODO: Flatten all parameters in a single class?


@dataclass
class Hellwig2022_Params:
    referenceLuminance: float
    L_A: float
    Y_b: float
    matrix_lms: ArrayLike
    compress_mode: bool


@dataclass
class Daniele_Evo_Params:
    peakLuminance: float
    n: float
    n_r: float
    g: float
    t_1: float
    c_t: float
    s_2: float
    u_2: float
    m_2: float


@dataclass
class ChromaCompression_Params:
    chroma_compress: float
    chroma_expand: float
    chroma_expand_thr: float
    cc_et: float
    limitJmax: float
    model_gamma: float
    sat: float
    sat_thr: float
    compr: float
    applyReachClamp: bool


@dataclass
class GamutCompression_Params:
    cuspMidBlend: float
    focusDistance: float
    focusDist: float
    focusAdjustGain: float
    focusGainBlend: float
    # TODO: Some of these parameters are duplicated in ChromaCompression
    limitJmax: float
    midJ: float
    model_gamma: float
    smoothCusps: float
    compressionFuncParams: ArrayLike
    reach_RGB_to_XYZ: ArrayLike
    reach_XYZ_to_RGB: ArrayLike


@dataclass
class Cusp_Params:
    tableSize: int
    gamutCuspTable: ArrayLike
    gamutCuspTableReach: ArrayLike
    cgamutCuspTable: ArrayLike
    cgamutReachTable: ArrayLike
    gamutTopGamma: ArrayLike
    gamutBottomGamma: float


@dataclass
class Input_Params:
    discountIlluminant: bool
    viewingConditions: str
    XYZ_to_RGB: ArrayLike
    RGB_to_XYZ: ArrayLike
    whiteXYZ: ArrayLike


@dataclass
class Limit_Params:
    discountIlluminant: bool
    viewingConditions: str
    XYZ_to_RGB: ArrayLike
    RGB_to_XYZ: ArrayLike
    whiteXYZ: ArrayLike


@dataclass
class Output_Params:
    discountIlluminant: bool
    viewingConditions: str
    XYZ_to_RGB: ArrayLike
    RGB_to_XYZ: ArrayLike
    eotf: Callable
    eotf_inverse: Callable
    fitWhite: bool
    softclamp: bool
    clamp_thr: float
    clamp_dist: float
    clamp: bool


def drt_params(
    inputDiscountIlluminant=True,
    inputViewingConditions="Dim",
    inputColourSpace=colour.models.RGB_COLOURSPACE_ACES2065_1,
    limitDiscountIlluminant=True,
    limitViewingConditions="Dim",
    limitColourSpace=colour.models.RGB_COLOURSPACE_sRGB,
    outputDiscountIlluminant=True,
    outputViewingConditions="Dim",
    outputColourSpace=colour.models.RGB_COLOURSPACE_sRGB,
):
    # LMS cone space for Hellwig 2022
    rxy = np.array([0.82, 0.175])
    gxy = np.array([-1.3, 1.8])
    bxy = np.array([0.13, -0.14])
    wxy = np.array([0.333, 0.333])
    CUSTOM_CAT16 = RGBPrimsToXYZMatrix(rxy, gxy, bxy, wxy, 1, 1)

    # Tone scale
    peakLuminanceTS = 100
    de_params = daniele_evo_params(peakLuminanceTS)

    # White
    white = np.array([1.0, 1.0, 1.0])
    inputWhiteXYZ = vector_dot(inputColourSpace.matrix_RGB_to_XYZ, white) * 100
    limitWhiteXYZ = vector_dot(limitColourSpace.matrix_RGB_to_XYZ, white) * 100

    params = {
        "Hellwig2022": Hellwig2022_Params(
            referenceLuminance=100,
            L_A=100,
            Y_b=20,
            matrix_lms=CUSTOM_CAT16,
            compress_mode=1,
        ),
        "Daniele_Evo": Daniele_Evo_Params(
            peakLuminance=de_params.peakLuminance,
            n=de_params.n,
            n_r=de_params.n_r,
            g=de_params.g,
            t_1=de_params.t_1,
            c_t=de_params.c_t,
            s_2=de_params.s_2,
            u_2=de_params.u_2,
            m_2=de_params.m_2,
        ),
        "ChromaCompression": ChromaCompression_Params(
            chroma_compress=3.5,
            chroma_expand=1.65,
            chroma_expand_thr=0.5,
            cc_et=3,
            limitJmax=None,
            model_gamma=None,
            sat=None,
            sat_thr=None,
            compr=None,
            applyReachClamp=None,
        ),
        "GamutCompression": GamutCompression_Params(
            cuspMidBlend=1.3,
            focusDistance=1.15,
            focusDist=None,
            focusAdjustGain=0.35,
            focusGainBlend=0.3,
            limitJmax=None,
            midJ=None,
            model_gamma=None,
            smoothCusps=0.26,
            compressionFuncParams=[0.75, 1.1, 1.3, 1.2],
            reach_RGB_to_XYZ=colour.models.RGB_COLOURSPACE_ACESCG.matrix_RGB_to_XYZ,
            reach_XYZ_to_RGB=colour.models.RGB_COLOURSPACE_ACESCG.matrix_XYZ_to_RGB,
        ),
        "Cusp": Cusp_Params(
            tableSize=360,
            gamutCuspTable=None,
            gamutCuspTableReach=None,
            cgamutCuspTable=None,
            cgamutReachTable=None,
            gamutTopGamma=None,
            gamutBottomGamma=1.15,
        ),
        "Input": Input_Params(
            discountIlluminant=inputDiscountIlluminant,
            viewingConditions=inputViewingConditions,
            XYZ_to_RGB=inputColourSpace.matrix_XYZ_to_RGB,
            RGB_to_XYZ=inputColourSpace.matrix_RGB_to_XYZ,
            whiteXYZ=inputWhiteXYZ,
        ),
        "Limit": Limit_Params(
            discountIlluminant=limitDiscountIlluminant,
            viewingConditions=limitViewingConditions,
            XYZ_to_RGB=limitColourSpace.matrix_XYZ_to_RGB,
            RGB_to_XYZ=limitColourSpace.matrix_RGB_to_XYZ,
            whiteXYZ=limitWhiteXYZ,
        ),
        "Output": Output_Params(
            discountIlluminant=outputDiscountIlluminant,
            viewingConditions=outputViewingConditions,
            XYZ_to_RGB=outputColourSpace.matrix_XYZ_to_RGB,
            RGB_to_XYZ=outputColourSpace.matrix_RGB_to_XYZ,
            eotf=outputColourSpace.cctf_decoding,
            eotf_inverse=outputColourSpace.cctf_encoding,
            fitWhite=False,
            softclamp=True,
            clamp_thr=0.9997,
            clamp_dist=1.1,
            clamp=True,
        ),
    }

    # Derived parameters

    hw_params = params["Hellwig2022"]
    de_params = params["Daniele_Evo"]
    cc_params = params["ChromaCompression"]
    gc_params = params["GamutCompression"]
    cusp_params = params["Cusp"]
    input_params = params["Input"]
    limit_params = params["Limit"]

    # Chroma Compression
    limitJmax = Y_to_J(
        peakLuminanceTS, hw_params.L_A, hw_params.Y_b, inputViewingConditions
    )

    surround = VIEWING_CONDITIONS_HELLWIG2022[inputViewingConditions]
    model_gamma = 1.0 / (surround.c * (1.48 + np.sqrt(hw_params.Y_b / hw_params.L_A)))
    log_peak = np.log10(de_params.n / de_params.n_r)
    compr = cc_params.chroma_compress + (cc_params.chroma_compress * 5.0) * log_peak
    sat = max(
        0.15, cc_params.chroma_expand - (cc_params.chroma_expand * 0.78) * log_peak
    )
    sat_thr = cc_params.chroma_expand_thr / de_params.n

    cc_params.limitJmax = limitJmax
    cc_params.model_gamma = model_gamma
    cc_params.sat = sat
    cc_params.sat_thr = sat_thr
    cc_params.compr = compr
    cc_params.applyReachClamp = False

    # Gamut Compression
    midJ = Y_to_J(
        de_params.c_t * de_params.n_r,
        hw_params.L_A,
        hw_params.Y_b,
        inputViewingConditions,
    )

    # Gamut mapper focus distance scaling with peak luminance for
    # HDR/SDR appearance match.  The projection gets slightly less
    # steep with higher peak luminance.
    # https:#www.desmos.com/calculator/bnfhjcq5vf
    focusDist = min(
        10.0, gc_params.focusDistance + gc_params.focusDistance * 1.65 * log_peak
    )

    gc_params.focusDist = focusDist
    gc_params.limitJmax = limitJmax
    gc_params.midJ = midJ
    gc_params.model_gamma = model_gamma

    # Cusp tables

    cusp_tables(
        input_params,
        hw_params,
        de_params,
        limit_params,
        cc_params,
        cusp_params,
        gc_params,
    )

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

    return Daniele_Evo_Params(
        peakLuminance,
        daniele_n,
        daniele_n_r,
        daniele_g,
        daniele_t_1,
        daniele_c_t,
        daniele_s_2,
        daniele_u_2,
        daniele_m_2,
    )


def cusp_tables(
    input_params,
    hellwig_params,
    devo_params,
    limit_params,
    cc_params,
    cusp_params,
    gc_params,
):

    boundaryRGB = devo_params.peakLuminance / hellwig_params.referenceLuminance

    h_samples = cusp_params.tableSize

    # Cusp table for chroma compression gamut

    gamutCuspTableUnsorted = np.zeros((h_samples, 3))
    h = np.linspace(0, 1, h_samples, endpoint=False)
    ones = np.ones(h.shape)
    RGB = HSV_to_RGB(tstack([h, ones, ones]))
    RGB *= boundaryRGB * hellwig_params.referenceLuminance
    gamutCuspTableUnsorted = luminance_RGB_to_JMh(
        RGB,
        limit_params.whiteXYZ,
        hellwig_params.L_A,
        hellwig_params.Y_b,
        input_params.viewingConditions,
        input_params.discountIlluminant,
        hellwig_params.matrix_lms,
        hellwig_params.compress_mode,
        gc_params.reach_RGB_to_XYZ,
    )
    minhIndex = np.argmin(gamutCuspTableUnsorted[..., 2], axis=-1)
    cgamutCuspTable = np.roll(gamutCuspTableUnsorted, -minhIndex, axis=0)

    # Reach table for the chroma compression reach.

    M_samples = 1300
    J = np.full((h_samples, M_samples), cc_params.limitJmax)
    M = np.arange(M_samples)
    M = np.tile(M, (h_samples, 1))
    h = np.arange(h_samples)
    h = np.tile(h, (M_samples, 1))
    h = np.moveaxis(h, 0, -1)
    JMhSearch = tstack([J, M, h])

    newLimitRGB = JMh_to_luminance_RGB(
        JMhSearch,
        limit_params.whiteXYZ,
        hellwig_params.L_A,
        hellwig_params.Y_b,
        input_params.viewingConditions,
        input_params.discountIlluminant,
        hellwig_params.matrix_lms,
        hellwig_params.compress_mode,
        gc_params.reach_XYZ_to_RGB,
    )
    newLimitRGB = newLimitRGB / boundaryRGB / hellwig_params.referenceLuminance

    zeros = np.zeros(newLimitRGB.shape)
    negatives = np.less(newLimitRGB, zeros)
    any_negatives = np.any(negatives, axis=-1)
    first_negative = any_negatives.argmax(axis=1)

    zeros = np.zeros((h_samples))
    h = np.arange(h_samples)
    cgamutReachTable = tstack([zeros, first_negative, h])

    # Cusp table for limiting gamut

    gamutCuspTableUnsorted = np.zeros((h_samples, 3))
    h = np.linspace(0, 1, h_samples, endpoint=False)
    ones = np.ones(h.shape)
    RGB = HSV_to_RGB(tstack([h, ones, ones]))
    RGB *= boundaryRGB * hellwig_params.referenceLuminance
    gamutCuspTableUnsorted = luminance_RGB_to_JMh(
        RGB,
        limit_params.whiteXYZ,
        hellwig_params.L_A,
        hellwig_params.Y_b,
        input_params.viewingConditions,
        input_params.discountIlluminant,
        hellwig_params.matrix_lms,
        hellwig_params.compress_mode,
        limit_params.RGB_to_XYZ,
    )
    minhIndex = np.argmin(gamutCuspTableUnsorted[..., 2], axis=-1)
    gamutCuspTable = np.roll(gamutCuspTableUnsorted, -minhIndex, axis=0)

    # Cusp table for limiting reach gamut.

    M_samples = 1300
    J = np.full((h_samples, M_samples), cc_params.limitJmax)
    M = np.arange(M_samples)
    M = np.tile(M, (h_samples, 1))
    h = np.arange(h_samples)
    h = np.tile(h, (M_samples, 1))
    h = np.moveaxis(h, 0, -1)
    JMhSearch = tstack([J, M, h])

    newLimitRGB = JMh_to_luminance_RGB(
        JMhSearch,
        limit_params.whiteXYZ,
        hellwig_params.L_A,
        hellwig_params.Y_b,
        input_params.viewingConditions,
        input_params.discountIlluminant,
        hellwig_params.matrix_lms,
        hellwig_params.compress_mode,
        gc_params.reach_XYZ_to_RGB,
    )
    newLimitRGB = newLimitRGB / boundaryRGB / hellwig_params.referenceLuminance

    zeros = np.zeros(newLimitRGB.shape)
    negatives = np.less(newLimitRGB, zeros)
    any_negatives = np.any(negatives, axis=-1)
    first_negative = any_negatives.argmax(axis=1)

    zeros = np.zeros((h_samples))
    h = np.arange(h_samples)
    gamutCuspTableReach = tstack([zeros, first_negative, h])

    # Upper hull gamma values for the gamut mapper
    # TODO: Vectorize the table derivation, sequential runs in more than 10sec.,
    # just use Nick's precomputed value for now.

    GT_gamutTopGamma = np.load("gamutTopGamma.npy")
    gamutTopGamma = GT_gamutTopGamma

    cusp_params.gamutCuspTable = gamutCuspTable
    cusp_params.gamutCuspTableReach = gamutCuspTableReach
    cusp_params.cgamutCuspTable = cgamutCuspTable
    cusp_params.cgamutReachTable = cgamutReachTable
    cusp_params.gamutTopGamma = gamutTopGamma

    # Compare table against Nick's Python reference

    GT_cgamutCuspTable = np.load("cgamutCuspTable.npy")
    np.testing.assert_almost_equal(cgamutCuspTable, GT_cgamutCuspTable)
    GT_cgamutReachTable = np.load("cgamutReachTable.npy")
    np.testing.assert_almost_equal(cgamutReachTable, GT_cgamutReachTable)
    GT_gamutCuspTable = np.load("gamutCuspTable.npy")
    np.testing.assert_almost_equal(gamutCuspTable, GT_gamutCuspTable)
    GT_gamutCuspTableReach = np.load("gamutCuspTableReach.npy")
    # Shuffle to restore JMh order
    tmp = GT_gamutCuspTableReach.copy()
    GT_gamutCuspTableReach[..., 0] = tmp[..., 1]
    GT_gamutCuspTableReach[..., 1] = tmp[..., 0]
    np.testing.assert_almost_equal(gamutCuspTableReach, GT_gamutCuspTableReach)
    np.testing.assert_almost_equal(gamutTopGamma, GT_gamutTopGamma)
