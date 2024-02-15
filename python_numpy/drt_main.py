import time

import colour
from colour.algebra import vector_dot
import numpy as np

np.set_printoptions(suppress=True)

from drt_init import drt_params
from drt_cam import (
    JMh_to_XYZ,
    XYZ_to_JMh,
)
from drt_tonescale import forwardTonescale, inverseTonescale
from drt_rgc import gamut_compression_operator
from drt_gamut_compress import gamutCompressForward, gamutCompressInverse


def aces_v2_drt(RGB, params):

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

    # # 1. Derive colour apparence correlates

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

    # if fitWhite:
    #     raise NotImplemented

    # Soft clamp by compressing negative display linear values
    # if softclampOutput:
    #     luminanceRGB = gamut_compression_operator(
    #         luminanceRGB,
    #         invert=False,
    #         threshold=[clamp_thr, clamp_thr, clamp_thr],
    #         cyan=clamp_dist,
    #         magenta=clamp_dist,
    #         yellow=clamp_dist,
    #         power=1.2
    #     )

    # Clamp to between zero and peak luminance
    if output_params.clamp:
        luminanceRGB = np.clip(luminanceRGB, 0, devo_params.peakLuminance)

    luminanceRGB = vector_dot(limit_params.RGB_to_XYZ, luminanceRGB)
    luminanceRGB = vector_dot(output_params.XYZ_to_RGB, luminanceRGB)
    luminanceRGB = luminanceRGB / hellwig_params.referenceLuminance

    outputRGB = output_params.eotf_inverse(luminanceRGB)
    outputRGB = np.clip(outputRGB, 0, 1)

    return outputRGB


def aces_v2_drt_inverse(RGB, params):

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

    XYZ = JMh_to_XYZ(
        JMh,
        input_params.whiteXYZ,
        hellwig_params.L_A,
        hellwig_params.Y_b,
        input_params.viewingConditions,
        input_params.discountIlluminant,
        matrix_lms=hellwig_params.matrix_lms,
        compress_mode=hellwig_params.compress_mode,
    )

    luminanceRGB = vector_dot(input_params.XYZ_to_RGB, XYZ)
    outputRGB = luminanceRGB / hellwig_params.referenceLuminance

    return outputRGB


if __name__ == "__main__":

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

    start = time.time()
    params = drt_params(
        inputDiscountIlluminant=True,
        inputViewingConditions="Dim",
        inputColourSpace=colour.models.RGB_COLOURSPACE_ACES2065_1,
        limitDiscountIlluminant=True,
        limitViewingConditions="Dim",
        limitColourSpace=BT1886_709_CS,
        outputDiscountIlluminant=True,
        outputViewingConditions="Dim",
        outputColourSpace=BT1886_709_CS,
    )
    end = time.time()
    print("Init params in", end - start)

    inRGB = colour.read_image("DigitalLAD.2048x1556.exr")[..., :3]
    # inRGB = np.array([
    #     [
    #         [0.18, 0.8, 0.8],
    #         [0.18, 0.8, 0.8],
    #     ],
    #     [
    #         [0.18, 0.8, 0.8],
    #         [0.18, 0.8, 0.8],
    #     ]
    # ])
    # inRGB = np.array([
    #     [0.18, 0.8, 0.8],
    #     [0.18, 0.8, 0.8],
    #     [0.18, 0.8, 0.8],
    #     [0.18, 0.8, 0.8],
    # ])
    # inRGB = np.array([0.18, 0.8, 0.8])

    # inRGB = inRGB[1556-1274-1, 692]
    # inRGB = inRGB[1556-1291-1, 688]

    start = time.time()
    RGB = aces_v2_drt(inRGB, params)
    end = time.time()
    print("Apply in", end - start)

    colour.write_image(RGB, "OutDigitalLAD.2048x1556.exr")

    # Round trip test
    # RGB = aces_v2_drt_inverse(inRGB, params)
    # RGB = aces_v2_drt(RGB, params)
    # np.testing.assert_almost_equal(RGB, inRGB)
