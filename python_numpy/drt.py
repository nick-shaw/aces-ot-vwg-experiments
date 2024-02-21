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

    hellwig_params = params["Hellwig2022"]
    devo_params = params["Daniele_Evo"]
    cc_params = params["ChromaCompression"]
    gc_params = params["GamutCompression"]
    cusp_params = params["Cusp"]
    input_params = params["Input"]
    limit_params = params["Limit"]
    output_params = params["Output"]

    # 0. Convert input to CIE XYZ

    HALF_MAX = np.finfo(np.half).max
    inputRGB = np.clip(RGB, -HALF_MAX, HALF_MAX)
    luminanceRGB = inputRGB * hellwig_params.referenceLuminance
    luminanceXYZ = vector_dot(input_params.RGB_to_XYZ, luminanceRGB)

    # 1. Derive colour apparence correlates

    JMh = XYZ_to_JMh(
        luminanceXYZ,
        input_params.whiteXYZ,
        hellwig_params.L_A,
        hellwig_params.Y_b,
        input_params.viewingConditions,
        input_params.discountIlluminant,
        matrix_lms=hellwig_params.matrix_lms,
        compress_mode=hellwig_params.compress_mode,
    )

    # 2. Apply tone scale on lightness attribute

    JMh = forwardTonescale(
        JMh,
        hellwig_params,
        devo_params,
        cc_params,
        cusp_params,
        limit_params,
        output_params,
    )

    # 3. Gamut compression

    JMh = gamutCompressForward(
        JMh,
        cusp_params,
        gc_params,
    )

    # 4. Convert to output colorimetry

    luminanceXYZ = JMh_to_XYZ(
        JMh,
        limit_params.whiteXYZ,
        hellwig_params.L_A,
        hellwig_params.Y_b,
        output_params.viewingConditions,
        output_params.discountIlluminant,
        matrix_lms=hellwig_params.matrix_lms,
        compress_mode=hellwig_params.compress_mode,
    )

    luminanceRGB = vector_dot(limit_params.XYZ_to_RGB, luminanceXYZ)

    # Clamp to between zero and peak luminance
    if output_params.clamp:
        luminanceRGB = np.clip(luminanceRGB, 0, devo_params.peakLuminance)

    luminanceRGB = vector_dot(limit_params.RGB_to_XYZ, luminanceRGB)
    luminanceRGB = vector_dot(output_params.XYZ_to_RGB, luminanceRGB)
    luminanceRGB = luminanceRGB / hellwig_params.referenceLuminance

    outputRGB = output_params.eotf_inverse(luminanceRGB)
    outputRGB = np.clip(outputRGB, 0, 1)

    return outputRGB


def drt_inverse(RGB, params):

    hellwig_params = params["Hellwig2022"]
    devo_params = params["Daniele_Evo"]
    cc_params = params["ChromaCompression"]
    gc_params = params["GamutCompression"]
    cusp_params = params["Cusp"]
    input_params = params["Input"]
    limit_params = params["Limit"]
    output_params = params["Output"]

    # 0. Convert input to CIE XYZ

    HALF_MAX = np.finfo(np.half).max
    inputRGB = np.clip(RGB, -HALF_MAX, HALF_MAX)
    luminanceRGB = output_params.eotf(inputRGB)
    luminanceRGB *= hellwig_params.referenceLuminance
    luminanceXYZ = vector_dot(output_params.RGB_to_XYZ, luminanceRGB)

    # 1. Derive colour apparence correlates

    JMh = XYZ_to_JMh(
        luminanceXYZ,
        limit_params.whiteXYZ,
        hellwig_params.L_A,
        hellwig_params.Y_b,
        output_params.viewingConditions,
        output_params.discountIlluminant,
        matrix_lms=hellwig_params.matrix_lms,
        compress_mode=hellwig_params.compress_mode,
    )

    # 2. Gamut un-compression

    JMh = gamutCompressInverse(
        JMh,
        cusp_params,
        gc_params,
    )

    # 3. Inverse tone scale on lightness attribute

    JMh = inverseTonescale(
        JMh,
        hellwig_params,
        devo_params,
        cc_params,
        cusp_params,
        limit_params,
        output_params,
    )

    # 4. Convert to input colorimetry

    luminanceXYZ = JMh_to_XYZ(
        JMh,
        input_params.whiteXYZ,
        hellwig_params.L_A,
        hellwig_params.Y_b,
        input_params.viewingConditions,
        input_params.discountIlluminant,
        matrix_lms=hellwig_params.matrix_lms,
        compress_mode=hellwig_params.compress_mode,
    )

    luminanceRGB = vector_dot(input_params.XYZ_to_RGB, luminanceXYZ)
    outputRGB = luminanceRGB / hellwig_params.referenceLuminance

    return outputRGB
