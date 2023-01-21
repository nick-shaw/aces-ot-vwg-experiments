#!/usr/bin/env python

"""
Hellwig and Fairchild (2022) Colour Appearance Model
====================================================

Defines the *Hellwig and Fairchild (2022)* colour appearance model objects:

-   :class:`colour.appearance.InductionFactors_Hellwig2022`
-   :attr:`colour.VIEWING_CONDITIONS_Hellwig2022`
-   :class:`colour.CAM_Specification_Hellwig2022`
-   :func:`colour.XYZ_to_Hellwig2022`
-   :func:`colour.Hellwig2022_to_XYZ`

References
----------
-   :cite:`Fairchild2022` : Fairchild, M. D., & Hellwig, L. (2022). Private
    Discussion with Mansencal, T.
-   :cite:`Hellwig2022` : Hellwig, L., & Fairchild, M. D. (2022). Brightness,
    lightness, colorfulness, and chroma in CIECAM02 and CAM16. Color Research
    & Application, col.22792. doi:10.1002/col.22792
"""

from __future__ import annotations

import colour
import matplotlib.pyplot as plt
import numpy as np
import itertools

import numpy as np
from collections import namedtuple
from dataclasses import astuple, dataclass, field

from colour.algebra import sdiv, sdiv_mode, spow, vector_dot, lerp
# from colour.appearance.cam16 import MATRIX_16, MATRIX_INVERSE_16
from colour.appearance.ciecam02 import (
    InductionFactors_CIECAM02,
    VIEWING_CONDITIONS_CIECAM02,
    hue_quadrature,
)
from colour.hints import (
    ArrayLike,
    Boolean,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    NDArray,
    Optional,
    Union,
)
from colour.utilities import (
    CanonicalMapping,
    MixinDataclassArithmetic,
    as_float,
    as_float_array,
    from_range_100,
    from_range_degrees,
    full,
    has_only_nan,
    ones,
    to_domain_100,
    to_domain_degrees,
    tsplit,
    tstack,
)
from colour.models import RGB_COLOURSPACE_sRGB

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "InductionFactors_Hellwig2022",
    "VIEWING_CONDITIONS_Hellwig2022",
    "CAM_Specification_Hellwig2022",
    "XYZ_to_Hellwig2022",
    "Hellwig2022_to_XYZ",
]

class InductionFactors_Hellwig2022(
    namedtuple("InductionFactors_Hellwig2022", ("F", "c", "N_c"))
):
    """
    *Hellwig and Fairchild (2022)* colour appearance model induction factors.

    Parameters
    ----------
    F
        Maximum degree of adaptation :math:`F`.
    c
        Exponential non-linearity :math:`c`.
    N_c
        Chromatic induction factor :math:`N_c`.

    Notes
    -----
    -   The *Hellwig and Fairchild (2022)* colour appearance model induction
        factors are the same as *CIECAM02* and *CAM16* colour appearance model.

    References
    ----------
    :cite:`Fairchild2022`, :cite:`Hellwig2022`
    """


VIEWING_CONDITIONS_Hellwig2022: CanonicalMapping = CanonicalMapping(
    VIEWING_CONDITIONS_CIECAM02
)
VIEWING_CONDITIONS_Hellwig2022.__doc__ = """
Reference *Hellwig and Fairchild (2022)* colour appearance model viewing
conditions.

References
----------
:cite:`Hellwig2022`
"""


@dataclass
class CAM_Specification_Hellwig2022(MixinDataclassArithmetic):
    """
    Define the *Hellwig and Fairchild (2022)* colour appearance model
    specification.

    Parameters
    ----------
    J
        Correlate of *Lightness* :math:`J`.
    C
        Correlate of *chroma* :math:`C`.
    h
        *Hue* angle :math:`h` in degrees.
    s
        Correlate of *saturation* :math:`s`.
    Q
        Correlate of *brightness* :math:`Q`.
    M
        Correlate of *colourfulness* :math:`M`.
    H
        *Hue* :math:`h` quadrature :math:`H`.
    HC
        *Hue* :math:`h` composition :math:`H^C`.

    References
    ----------
    :cite:`Fairchild2022`, :cite:`Hellwig2022`
    """

    J: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    C: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    h: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    s: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    Q: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    M: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    H: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)
    HC: Optional[FloatingOrNDArray] = field(default_factory=lambda: None)


def f_RGB_a(RGB: ArrayLike, F_L: FloatingOrArrayLike) -> NDArray:
    F_L_RGB = (F_L[..., np.newaxis] * np.absolute(RGB) / 100) ** 0.42
    RGB_a = (400 * np.sign(RGB) * F_L_RGB) / (27.13 + F_L_RGB) + 0.1

    return RGB_a


def f_i_RGB_a(RGB: ArrayLike, F_L: FloatingOrArrayLike) -> NDArray:
    RGB_p = (
        np.sign(RGB - 0.1)
        * 100
        / F_L[..., np.newaxis]
        * ((27.13 * np.absolute(RGB - 0.1)) / (400 - np.absolute(RGB - 0.1))) ** (1 / 0.42)
    )

    return RGB_p


def f_d_RGB_a(RGB: ArrayLike, F_L: FloatingOrArrayLike) -> NDArray:
    F_L_RGB = spow(F_L[..., np.newaxis] * RGB / 100, 0.42)
    F_L_100 = spow(F_L[..., np.newaxis] / 100, 0.42)

    d_RGB_a = (
        400 * ((0.42 * 27.13) * spow(RGB, -0.58) * F_L_100) / (F_L_RGB + 27.13) ** 2
    )

    return d_RGB_a


def compress(xyz):
    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]

    C = (x+y+z)/3
    R = np.sqrt((x-C)**2 + (y-C)**2 + (z-C)**2)
    R *= np.sqrt(2/3)

    x = (x-C)/R 
    y = (y-C)/R 
    z = (z-C)/R 

    r = R/C
    s = -np.minimum(x, np.minimum(y, z))

    t = 1/(0.5+((s-0.5)**2 + (np.sqrt(4/r**2+1)-1)**2/4)**0.5)

    x = C*x*t + C
    y = C*y*t + C
    z = C*z*t + C  

    RRR = tstack((R, R, R))
    return np.where(RRR==0.0, xyz, np.array([x,y,z]).T)


def uncompress(xyz):
    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]

    C = (x+y+z)/3
    R = np.sqrt((x-C)**2 + (y-C)**2 + (z-C)**2)
    R *= np.sqrt(2/3)

    x = (x-C)/R 
    y = (y-C)/R 
    z = (z-C)/R 

    t = R/C
    s = -np.minimum(x, np.minimum(y, z))

    r = 2/np.sqrt((2*np.sqrt((1/t-0.5)**2-(s-0.5)**2)+1)**2-1)

    x = C*x*r + C
    y = C*y*r + C
    z = C*z*r + C  

    RRR = tstack((R, R, R))
    return np.where(RRR==0.0, xyz, np.array([x,y,z]).T)


# LINEAR_VS_COMPRESS = False
# DO_COMPRESS = False
# DO_ECCENTRICITY = False

def XYZ_to_Hellwig2022(
    XYZ: ArrayLike,
    XYZ_w: ArrayLike,
    L_A: FloatingOrArrayLike,
    Y_b: FloatingOrArrayLike,
    surround: Union[
        InductionFactors_CIECAM02, InductionFactors_Hellwig2022
    ] = VIEWING_CONDITIONS_Hellwig2022["Average"],
    L_B=0,
    H_B=1,
    discount_illuminant: Boolean = False,
) -> CAM_Specification_Hellwig2022:
    XYZ = to_domain_100(XYZ)
    XYZ_w = to_domain_100(XYZ_w)
    _X_w, Y_w, _Z_w = tsplit(XYZ_w)
    L_A = as_float_array(L_A)
    Y_b = as_float_array(Y_b)

    # Step 0
    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    RGB_w = vector_dot(MATRIX_16, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = (
        np.clip(surround.F * (1 - (1 / 3.6) * np.exp((-L_A - 42) / 92)), 0, 1)
        if not discount_illuminant
        else ones(L_A.shape)
    )

    # Viewing conditions dependent parameters
    k = 1 / (5 * L_A + 1)
    k4 = k**4
    F_L = 0.2 * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * spow(5 * L_A, 1 / 3)
    n = sdiv(Y_b, Y_w)
    z = 1.48 + np.sqrt(n)

    D_RGB = D[..., np.newaxis] * Y_w[..., np.newaxis] / RGB_w + 1 - D[..., np.newaxis]
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non-linear response compression.
    F_L_RGB_w = spow(F_L[..., np.newaxis] * np.absolute(RGB_wc) / 100, 0.42)
    RGB_aw = (400 * np.sign(RGB_wc) * F_L_RGB_w) / (27.13 + F_L_RGB_w) + 0.1

    # Computing achromatic responses for the whitepoint.
    R_aw, G_aw, B_aw = tsplit(RGB_aw)
    A_w = 2 * R_aw + G_aw + 0.05 * B_aw - 0.305

    # Step 1
    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    RGB = vector_dot(MATRIX_16, XYZ)

    # Step 2
    RGB_c = D_RGB * RGB

    # Step 3

    # # Applying forward post-adaptation non-linear response compression.
    if LINEAR_VS_COMPRESS:
        RGB_a = f_RGB_a(RGB_c, F_L)
        RGB_a_l = f_d_RGB_a(full(3, L_B), F_L) * (RGB_c - L_B) + f_RGB_a(full(3, L_B), F_L)
        # RGB_a[RGB_c < L_B] = 0
        # RGB_a = np.where(np.logical_or(RGB_c < L_B, RGB_c > H_B), RGB_a_l, RGB_a)
        RGB_a = np.where(RGB_c < L_B, RGB_a_l, RGB_a)
    else:
        if DO_COMPRESS:
            RGB_c = compress(RGB_c)
        F_L_RGB = spow(F_L[..., np.newaxis] * np.absolute(RGB_c) / 100, 0.42)
        RGB_a = (400 * np.sign(RGB_c) * F_L_RGB) / (27.13 + F_L_RGB) + 0.1
        if DO_COMPRESS:
            RGB_a = uncompress(RGB_a)

    # Step 4
    # Converting to preliminary cartesian coordinates.
    R_a, G_a, B_a = tsplit(RGB_a)
    a = R_a - 12 * G_a / 11 + B_a / 11
    b = (R_a + G_a - 2 * B_a) / 9

    # Computing the *hue* angle :math:`h`.
    h = np.degrees(np.arctan2(b, a)) % 360

    # Step 5
    # Computing eccentricity factor *e_t*.
    hr = np.radians(h)

    _h = hr
    _2_h = 2 * hr
    _3_h = 3 * hr
    _4_h = 4 * hr

    e_t = (
        -0.0582 * np.cos(_h)
        - 0.0258 * np.cos(_2_h)
        - 0.1347 * np.cos(_3_h)
        + 0.0289 * np.cos(_4_h)
        - 0.1475 * np.sin(_h)
        - 0.0308 * np.sin(_2_h)
        + 0.0385 * np.sin(_3_h)
        + 0.0096 * np.sin(_4_h)
        + 1
    )
    if not DO_ECCENTRICITY:
        e_t = 1.0

    # Computing hue :math:`h` quadrature :math:`H`.
    H = hue_quadrature(h)
    # TODO: Compute hue composition.

    # Step 6
    # Computing achromatic responses for the stimulus.
    R_a, G_a, B_a = tsplit(RGB_a)
    A = 2 * R_a + G_a + 0.05 * B_a - 0.305

    # Step 7
    # Computing the correlate of *Lightness* :math:`J`.
    with sdiv_mode():
        J = 100 * spow(sdiv(A, A_w), surround.c * z)

    # Step 8
    # Computing the correlate of *brightness* :math:`Q`.
    with sdiv_mode():
        Q = (2 / as_float(surround.c)) * (J / 100) * A_w

    # Step 9
    # Computing the correlate of *colourfulness* :math:`M`.
    M = 43 * surround.N_c * e_t * np.sqrt(a**2 + b**2)

    # Computing the correlate of *chroma* :math:`C`.
    with sdiv_mode():
        C = 35 * sdiv(M, A_w)

    # Computing the correlate of *saturation* :math:`s`.
    with sdiv_mode():
        s = 100 * sdiv(M, Q)

    return CAM_Specification_Hellwig2022(
        as_float(from_range_100(J)),
        as_float(from_range_100(C)),
        as_float(from_range_degrees(h)),
        as_float(from_range_100(s)),
        as_float(from_range_100(Q)),
        as_float(from_range_100(M)),
        as_float(from_range_degrees(H, 400)),
        None,
    )


def Hellwig2022_to_XYZ(
    specification: CAM_Specification_Hellwig2022,
    XYZ_w: ArrayLike,
    L_A: FloatingOrArrayLike,
    Y_b: FloatingOrArrayLike,
    surround: Union[
        InductionFactors_CIECAM02, InductionFactors_Hellwig2022
    ] = VIEWING_CONDITIONS_Hellwig2022["Average"],
    L_B=0,
    H_B=1,
    discount_illuminant: Boolean = False,
) -> NDArray:
    J, C, h, _s, _Q, M, _H, _HC = astuple(specification)

    J = to_domain_100(J)
    C = to_domain_100(C)
    h = to_domain_degrees(h)
    M = to_domain_100(M)
    L_A = as_float_array(L_A)
    XYZ_w = to_domain_100(XYZ_w)
    _X_w, Y_w, _Z_w = tsplit(XYZ_w)

    # Step 0
    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    RGB_w = vector_dot(MATRIX_16, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = (
        np.clip(surround.F * (1 - (1 / 3.6) * np.exp((-L_A - 42) / 92)), 0, 1)
        if not discount_illuminant
        else ones(L_A.shape)
    )

    # Viewing conditions dependent parameters
    k = 1 / (5 * L_A + 1)
    k4 = k**4
    F_L = 0.2 * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * spow(5 * L_A, 1 / 3)
    n = sdiv(Y_b, Y_w)
    z = 1.48 + np.sqrt(n)

    D_RGB = D[..., np.newaxis] * Y_w[..., np.newaxis] / RGB_w + 1 - D[..., np.newaxis]
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non-linear response compression.
    F_L_RGB_w = spow(F_L[..., np.newaxis] * np.absolute(RGB_wc) / 100, 0.42)
    RGB_aw = (400 * np.sign(RGB_wc) * F_L_RGB_w) / (27.13 + F_L_RGB_w) + 0.1

    # Computing achromatic responses for the whitepoint.
    R_aw, G_aw, B_aw = tsplit(RGB_aw)
    A_w = 2 * R_aw + G_aw + 0.05 * B_aw - 0.305

    # Step 1
    if has_only_nan(M) and not has_only_nan(C):
        M = (C * A_w) / 35
    elif has_only_nan(M):
        raise ValueError(
            'Either "C" or "M" correlate must be defined in '
            'the "CAM_Specification_Hellwig2022" argument!'
        )

    # Step 2
    # Computing eccentricity factor *e_t*.
    hr = np.radians(h)

    _h = hr
    _2_h = 2 * hr
    _3_h = 3 * hr
    _4_h = 4 * hr

    e_t = (
        -0.0582 * np.cos(_h)
        - 0.0258 * np.cos(_2_h)
        - 0.1347 * np.cos(_3_h)
        + 0.0289 * np.cos(_4_h)
        - 0.1475 * np.sin(_h)
        - 0.0308 * np.sin(_2_h)
        + 0.0385 * np.sin(_3_h)
        + 0.0096 * np.sin(_4_h)
        + 1
    )
    if not DO_ECCENTRICITY:
        e_t = 1.0

    # Computing achromatic response :math:`A` for the stimulus.
    A = A = A_w * spow(J / 100, 1 / (surround.c * z))

    # Computing *P_p_1* to *P_p_2*.
    P_p_1 = 43 * surround.N_c * e_t
    P_p_2 = A

    # Step 3
    # Computing opponent colour dimensions :math:`a` and :math:`b`.
    with sdiv_mode():
        gamma = M / P_p_1

    a = gamma * np.cos(hr)
    b = gamma * np.sin(hr)

    # Step 4
    # Applying post-adaptation non-linear response compression matrix.
    RGB_a = (
        vector_dot(
            [
                [460, 451, 288],
                [460, -891, -261],
                [460, -220, -6300],
            ],
            tstack([P_p_2, a, b]),
        )
        / 1403
    )

    # Step 5
    # Applying inverse post-adaptation non-linear response compression.
    if LINEAR_VS_COMPRESS:
        RGB_c = f_i_RGB_a(RGB_a + 0.1, F_L)
        RGB_c_l = (RGB_a + 0.1 - f_RGB_a(full(3, L_B), F_L)) / (
            f_d_RGB_a(full(3, L_B), F_L)
        ) + L_B
        # RGB_c[RGB_a < L_B] = 0
        # RGB_c = np.where(np.logical_or(RGB_c < L_B, RGB_c > H_B), RGB_c_l, RGB_c)
        RGB_c = np.where(RGB_c < L_B, RGB_c_l, RGB_c)
    else:
        RGB_a = compress(RGB_a)
        RGB_c = (
            np.sign(RGB_a)
            * 100
            / F_L[..., np.newaxis]
            * spow(
                (27.13 * np.absolute(RGB_a)) / (400 - np.absolute(RGB_a)),
                1 / 0.42,
            )
        )
        RGB_c = uncompress(RGB_c)

    # Step 6
    RGB = RGB_c / D_RGB

    # Step 7
    XYZ = vector_dot(MATRIX_INVERSE_16, RGB)

    return from_range_100(XYZ)

def vec_length(vec2):
    a, b = tsplit(vec2)
    return np.sqrt(a * a + b * b)

def vec_normalize(vec2):
    l = vec_length(vec2)
    a, b = tsplit(vec2)
    return np.array([a / l, b / l]).T

def compress_powerp(v, threshold, limit, power):
    s = (limit - threshold) / np.power(np.power((1.0 - threshold) / (limit - threshold), -power) - 1.0, 1.0 / power)
    if v < threshold or limit < 1.0001:
        return v
    return threshold + s * ((v - threshold) / s) / (np.power(1.0 + np.power((v - threshold) / s, power), 1.0 / power))

# CAM DRT gamut boundary intersection finder
def find_gamut_intersection(pfrom, focus, h, precision = 10, startStepSize = 0.4):
    fromJ, fromM = tsplit(pfrom)
    focusJ, focusM = tsplit(focus)

    Jdiff = fromJ - focusJ
    project_to = np.array([focusJ - ((Jdiff / (fromM - focusM)) * focusM), 0.0]).T
    toJ, toM = tsplit(project_to)
    if toJ <= 0.0 or toJ >= 100:
        return project_to

    XYZ_w = colour.xy_to_XYZ(PLOT_COLOURSPACE.whitepoint) * 100
    L_A = 100
    Y_b = 20
    surround = colour.VIEWING_CONDITIONS_HELLWIG2022["Dim"]

    stepSize = startStepSize
    unitVector = vec_normalize(np.subtract(project_to, focus));
    JMtest = project_to
    searchOutwards = True

    for i in range(precision):
        for k in range(30):
            JMtest = np.add(JMtest, unitVector * stepSize)
            testJ, testM = tsplit(JMtest)

            JMh = CAM_Specification_Hellwig2022(J=testJ, M=testM, h=h)
            XYZ = Hellwig2022_to_XYZ(JMh, XYZ_w, L_A, Y_b, surround, discount_illuminant=True)
            RGB = vector_dot(PLOT_COLOURSPACE.matrix_XYZ_to_RGB, XYZ) / 100

            neg = RGB.min() < 0
            over = RGB.max() > 1
            nan = np.isnan(XYZ.min())
            out = np.logical_or(nan, np.logical_or(neg, over))

            if searchOutwards:
                if testJ < 0.0 or testJ > 100 or testM > 100 or out:
                    searchOutwards = False
                    stepSize = -abs(stepSize) / 2.0
                    break
            elif testM < 0.0 or out == 0:
                searchOutwards = True
                stepSize = abs(stepSize) / 2.0
                break

    return JMtest

# Gamut intersection to triangle (TODO: smoothing triangle cusp and curved shape)
def find_gamut_intersection_tri(cusp, pfrom, pto):
    cuspJ, cuspM = tsplit(cusp)
    fromJ, fromM = tsplit(pfrom)
    toJ, toM = tsplit(pto)

    if (fromJ - toJ) * cuspM - (cuspJ - toJ) * fromM <= 0.0:
        t = cuspM * toJ / (fromM * cuspJ + cuspM * (toJ - fromJ))
    else:
        t = cuspM * (toJ - 100) / (fromM * (cuspJ - 100) + cuspM * (toJ - fromJ))

    return np.array([toJ * (1.0 - t) + t * fromJ, t * fromM]).T

def forwardGamutMapper(JMh, cusp, approx):
    J, M, h = tsplit(JMh)
    cuspJ, cuspM = tsplit(cusp)

    if M == 0.0:
        return JMh

    # Calculate where the out of gamut color is projected to
    # Mid gray assumed to be 10.0 bits (34.0 J)
    focusJ = lerp(0.5, cuspJ, 34.0)
    Jdiff = J - focusJ
    if Jdiff > 0.0:
        focusDistanceGain = (100 - focusJ) / np.maximum(0.0001, 100 - np.minimum(100, J))
    else:
        focusDistanceGain = focusJ / np.maximum(0.0001, J)

    # in v28 this depends on tonescaled lightness so just set it to the old 0.5 value
    focusAdjust = 0.5
    focusM = -cuspM * focusAdjust * focusDistanceGain
    projectJ = focusJ - ((Jdiff / (M - focusM)) * focusM)

    # Find gamut intersection
    project_from = np.array([J, M]).T
    project_to = np.array([projectJ, 0]).T
    if approx:
        boundary = find_gamut_intersection_tri(cusp, project_from, project_to)
    else:
        boundary = find_gamut_intersection(project_from, np.array([focusJ, focusM]).T, h)

    # Compress
    normFact = 1.0 / np.maximum(0.0001, vec_length(np.subtract(boundary, project_to)))
    v = vec_length(np.subtract(project_from, project_to)) * normFact
    # In v28 limit depends on tonescaled lightness so set it to the old 1.2 value
    vc = compress_powerp(v, 0.75, 1.2, 1.2)
    compressed = np.add(project_to, vec_normalize(np.subtract(project_from, project_to)) * vc / normFact)

    J, M = tsplit(compressed)
    return np.array([J, M, h, projectJ, boundary[...,0], boundary[...,1]]).T

def find_threshold(J, h, iterations=10, debug=False):
    XYZ_w = colour.xy_to_XYZ(PLOT_COLOURSPACE.whitepoint) * 100
    L_A = 100
    Y_b = 20
    surround = colour.VIEWING_CONDITIONS_HELLWIG2022["Dim"]
    M_max = np.minimum(J*1.25+25, 100.0)
    M = tstack((np.zeros(len(J)), M_max))
#     M = np.array([0, min(J*1.25+25, 100.0)])
    i = iterations
    while i >= 0:
        mean = M.mean(axis=1)
        JMh = CAM_Specification_Hellwig2022(J=J, M=mean, h=h)
        XYZ = Hellwig2022_to_XYZ(JMh, XYZ_w, L_A, Y_b, surround, discount_illuminant=True)
        RGB = vector_dot(PLOT_COLOURSPACE.matrix_XYZ_to_RGB, XYZ) / 100
        M0, M1 = tsplit(M)
        neg = RGB.min(axis=1) < 0
        over = RGB.max(axis=1) > 1
        nan = np.isnan(XYZ.min(axis=1))
        out_of_gamut = np.logical_or(nan, np.logical_or(neg, over))
        if debug:
            print('M:')
            print(M)
            print('XYZ:')
            print(XYZ)
            print('RGB:')
            print(RGB)
            print('Bools:')
            print(tstack((neg, over, nan)))
        M1_new = np.where(out_of_gamut, mean, M1)
        M0_new = np.where(out_of_gamut, M0, mean)
        M = tstack((M0_new, M1_new))
#         if RGB.min() < 0 or RGB.max() > 1 or np.isnan(XYZ.min()):
#             M[1] = M.mean()
#         else:
#             M[0] = M.mean()
        i -= 1
    return M.mean(axis=1)

def find_boundary(h, iterations=10):
    J_range = np.linspace(0, 100, J_resolution)
#     M_boundary = np.zeros(J_resolution)
#     j = 0
    M_boundary = find_threshold(J_range, h, iterations)
#     for J in J_range: 
#         M_boundary[j] = find_threshold(J, h, iterations)
#         j += 1
        
    return M_boundary


PLOT_COLOURSPACE = colour.models.RGB_COLOURSPACE_BT709
space_name = "BT709" # filename friendly name

J_resolution = 1024 # samples along J axis
iterations = 10 # number of iterations for boundary solve

# Custom LMS Primaries from DRT v28
RGB_COLOURSPACE_MATRIX_16 = colour.RGB_Colourspace(
    "MATRIX 16",
    np.array(
        [
            [0.84, 0.175],
            [-1.3, 1.75],
            [0.05, -0.15],
        ]
    ),
    [4200, -1050],
)

MATRIX_16 = RGB_COLOURSPACE_MATRIX_16.matrix_XYZ_to_RGB
MATRIX_INVERSE_16 = np.linalg.inv(MATRIX_16)

LINEAR_VS_COMPRESS = False
DO_COMPRESS = True
DO_ECCENTRICITY = True

if __name__ == "__main__":
    import os
    if not os.path.exists('./data'):
        os.makedirs('./data')

    M_cusp = np.zeros(360)
    J_cusp = np.zeros(360)
    for h in range(360):
#        if h % 10 == 0:
#            print(h)
        M_boundary = find_boundary(h*1.0, iterations=iterations)
        M_cusp[h] = M_boundary.max()
        J_cusp[h] = 100.0 * (M_boundary.argmax()) / (J_resolution - 1)

    np.savetxt('./data/M_cusp_{}.txt'.format(space_name), M_cusp)
    np.savetxt('./data/J_cusp_{}.txt'.format(space_name), J_cusp)
    plt.plot(np.linspace(0, 359, 360), M_cusp, label='M cusp')
    plt.plot(np.linspace(0, 359, 360), J_cusp, label='J cusp')
    plt.ylim(0, 100)
    XYZ_w = colour.xy_to_XYZ(PLOT_COLOURSPACE.whitepoint) * 100
    L_A = 100.0
    Y_b = 20.0
    surround = colour.VIEWING_CONDITIONS_HELLWIG2022["Dim"]
    rgbcmyk = [['red', [1.0, 0.0, 0.0]],
               ['green', [0.0, 1.0, 0.0]],
               ['blue', [0.0, 0.0, 1.0]],
               ['cyan', [0.0, 1.0, 1.0]],
               ['magenta', [1.0, 0.0, 1.0]],
               ['yellow', [1.0, 1.0, 0.0]]]
    for cname, RGB in rgbcmyk:
        XYZ = vector_dot(PLOT_COLOURSPACE.matrix_RGB_to_XYZ, np.array(RGB)*100)
        hellwig = XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround, discount_illuminant=True)
        J = hellwig.J
        M = hellwig.M
        h = hellwig.h
        plt.scatter(h, J, c=cname)
        plt.scatter(h, M, c=cname)
    plt.xlabel('Hellwig h')
    plt.ylabel('Cusp value')
    plt.title('Cusp Paths {}'.format(PLOT_COLOURSPACE.name))
    plt.legend()
    plt.savefig('cusp_paths_{}.png'.format(space_name))
    plt.show()
