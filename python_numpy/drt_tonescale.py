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

from drt_cam import (
    JMh_to_luminance_RGB,
    luminance_RGB_to_JMh,
)
from drt_chroma_compress import (
    chromaCompressionForward,
    chromaCompressionInverse,
)


def forwardTonescale(
    inputJMh: ArrayLike,
    hellwig_params,
    devo_params,
    cc_params,
    cusp_params,
    limit_params,
    output_params,
) -> NDArrayFloat:

    J, M, h = tsplit(inputJMh)
    zeros = np.zeros(J.shape)
    monoJMh = tstack([J, zeros, zeros])

    linearJMh = JMh_to_luminance_RGB(
        monoJMh,
        limit_params.whiteXYZ,
        hellwig_params.L_A,
        hellwig_params.Y_b,
        limit_params.viewingConditions,
        limit_params.discountIlluminant,
        hellwig_params.matrix_lms,
        hellwig_params.compress_mode,
        output_params.XYZ_to_RGB,
    )

    linear = linearJMh[..., 0] / hellwig_params.referenceLuminance
    luminanceTS = daniele_evo_fwd(linear, devo_params)
    luminanceTS *= devo_params.n_r

    tonemappedmonoJMh = luminance_RGB_to_JMh(
        tstack([luminanceTS, luminanceTS, luminanceTS]),
        limit_params.whiteXYZ,
        hellwig_params.L_A,
        hellwig_params.Y_b,
        limit_params.viewingConditions,
        limit_params.discountIlluminant,
        hellwig_params.matrix_lms,
        hellwig_params.compress_mode,
        output_params.RGB_to_XYZ,
    )
    tonemappedJMh = tstack([tonemappedmonoJMh[..., 0], M, h])

    outputJMh = tonemappedJMh

    outputJMh[..., 1] = chromaCompressionForward(
        tonemappedJMh, J, cc_params, cusp_params
    )

    return outputJMh


def inverseTonescale(
    JMh: ArrayLike,
    hellwig_params,
    devo_params,
    cc_params,
    cusp_params,
    limit_params,
    output_params,
) -> NDArrayFloat:

    J, M, h = tsplit(JMh)
    zeros = np.zeros(J.shape)
    monoTonemappedJMh = tstack([J, zeros, zeros])

    monoTonemappedRGB = JMh_to_luminance_RGB(
        monoTonemappedJMh,
        limit_params.whiteXYZ,
        hellwig_params.L_A,
        hellwig_params.Y_b,
        limit_params.viewingConditions,
        limit_params.discountIlluminant,
        hellwig_params.matrix_lms,
        hellwig_params.compress_mode,
        output_params.XYZ_to_RGB,
    )

    luminance = monoTonemappedRGB[..., 0]
    luminance /= devo_params.n_r
    linear = daniele_evo_rev(luminance, devo_params)
    linear = linear * hellwig_params.referenceLuminance

    untonemappedMonoJMh = luminance_RGB_to_JMh(
        tstack([linear, linear, linear]),
        limit_params.whiteXYZ,
        hellwig_params.L_A,
        hellwig_params.Y_b,
        limit_params.viewingConditions,
        limit_params.discountIlluminant,
        hellwig_params.matrix_lms,
        hellwig_params.compress_mode,
        output_params.RGB_to_XYZ,
    )
    untonemappedColourJMh = tstack([untonemappedMonoJMh[..., 0], M, h])

    untonemappedColourJMh[..., 1] = chromaCompressionInverse(
        JMh, untonemappedMonoJMh[..., 0], cc_params, cusp_params
    )

    return untonemappedColourJMh


def daniele_evo_fwd(Y, params):
    Y = np.maximum(Y, np.zeros(Y.shape)) / (Y + params.s_2)
    f = params.m_2 * np.power(Y, params.g)
    h = np.maximum(f * f / (f + params.t_1), np.zeros(Y.shape))

    return h


def daniele_evo_rev(Y, params):
    Y = np.maximum(
        np.minimum(params.n / (params.u_2 * params.n_r), Y), np.zeros(Y.shape)
    )
    h = (Y + np.sqrt(Y * (4.0 * params.t_1 + Y))) / 2.0
    f = params.s_2 / (np.power((params.m_2 / h), (1.0 / params.g)) - 1.0)

    return f
