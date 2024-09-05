
import colour
from colour.algebra import vector_dot

import numpy as np

np.set_printoptions(precision=12, suppress=True)

from drt_cam import (
    JMh_to_XYZ,
    XYZ_to_JMh,
)
from drt_tonescale import forwardTonescale, inverseTonescale
from drt_gamut_compress import gamutCompressForward, gamutCompressInverse



def drt_forward(RGB, params):

    # 0. Convert input to CIE XYZ

    luminanceRGB = RGB * params.referenceLuminance
    luminanceXYZ = vector_dot(params.input_RGB_to_XYZ, luminanceRGB)

    if params.ap1_clamp:
        luminanceRGB = vector_dot(colour.models.RGB_COLOURSPACE_ACESCG.matrix_XYZ_to_RGB, luminanceXYZ)
        luminanceRGB = np.clip(luminanceRGB, 0, np.finfo(np.float32).max)
        luminanceXYZ = vector_dot(colour.models.RGB_COLOURSPACE_ACESCG.matrix_RGB_to_XYZ, luminanceRGB)

    # 1. Derive colour apparence correlates

    JMh = XYZ_to_JMh(
        luminanceXYZ,
        params.input_whiteXYZ,
        params.L_A,
        params.Y_b,
        params.input_viewingConditions,
        params.input_discountIlluminant,
        matrix_lms=params.matrix_lms,
    )

    # 2. Apply tone scale on lightness attribute

    tonemappedJMh = forwardTonescale(JMh, params)

    # 3. Gamut compression

    compressedJMh = gamutCompressForward(tonemappedJMh, params)

    # 4. Convert to output colorimetry

    luminanceXYZ = JMh_to_XYZ(
        compressedJMh,
        params.limit_whiteXYZ,
        params.L_A,
        params.Y_b,
        params.output_viewingConditions,
        params.output_discountIlluminant,
        matrix_lms=params.matrix_lms,
    )

    luminanceRGB = vector_dot(params.limit_XYZ_to_RGB, luminanceXYZ)

    # Clamp to between zero and peak luminance
    if params.clamp:
        luminanceRGB = np.clip(luminanceRGB, 0, params.peakLuminance)

    luminanceRGB = luminanceRGB / params.referenceLuminance
    luminanceRGB = vector_dot(params.limit_RGB_to_XYZ, luminanceRGB)
    luminanceRGB = vector_dot(params.output_XYZ_to_RGB, luminanceRGB)

    outputRGB = params.eotf_inverse(luminanceRGB)
    outputRGB = np.clip(outputRGB, 0, 1)

    return outputRGB


def drt_inverse(RGB, params):

    # 0. Convert input to CIE XYZ

    luminanceRGB = params.eotf(RGB)
    luminanceRGB *= params.referenceLuminance
    luminanceXYZ = vector_dot(params.output_RGB_to_XYZ, luminanceRGB)

    # 1. Derive colour apparence correlates

    compressedJMh = XYZ_to_JMh(
        luminanceXYZ,
        params.limit_whiteXYZ,
        params.L_A,
        params.Y_b,
        params.output_viewingConditions,
        params.output_discountIlluminant,
        matrix_lms=params.matrix_lms,
    )

    # 2. Gamut un-compression

    tonemappedJMh = gamutCompressInverse(compressedJMh, params)

    # 3. Inverse tone scale on lightness attribute

    JMh = inverseTonescale(tonemappedJMh, params)

    # 4. Convert to input colorimetry

    luminanceXYZ = JMh_to_XYZ(
        JMh,
        params.input_whiteXYZ,
        params.L_A,
        params.Y_b,
        params.input_viewingConditions,
        params.input_discountIlluminant,
        matrix_lms=params.matrix_lms,
    )

    luminanceRGB = vector_dot(params.input_XYZ_to_RGB, luminanceXYZ)
    outputRGB = luminanceRGB / params.referenceLuminance

    return outputRGB
