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
    Hellwig_J_to_Y,
    Y_to_Hellwig_J,
)
from drt_chroma_compress import (
    chromaCompressionForward,
    chromaCompressionInverse,
)


def forwardTonescale(inputJMh: ArrayLike, params) -> NDArrayFloat:

    J, M, h = tsplit(inputJMh)

    linear = Hellwig_J_to_Y(
        J,
        params.L_A,
        params.Y_b,
        params.limit_viewingConditions) / params.referenceLuminance
    luminanceTS = daniele_evo_fwd(linear, params)
    luminanceTS *= params.n_r

    tonemappedJ = Y_to_Hellwig_J(
        luminanceTS,
        params.L_A,
        params.Y_b,
        params.limit_viewingConditions
    )
    tonemappedJMh = tstack([tonemappedJ, M, h])

    outputJMh = tonemappedJMh
    outputJMh[..., 1] = chromaCompressionForward(
        tonemappedJMh, J, params
    )

    return outputJMh


def inverseTonescale(JMh: ArrayLike, params) -> NDArrayFloat:

    J, M, h = tsplit(JMh)

    luminance = Hellwig_J_to_Y(
        J,
        params.L_A,
        params.Y_b,
        params.limit_viewingConditions
    )

    luminance /= params.n_r
    linear = daniele_evo_rev(luminance, params)
    linear = linear * params.referenceLuminance

    untonemappedJ = Y_to_Hellwig_J(
        linear,
        params.L_A,
        params.Y_b,
        params.limit_viewingConditions
    )
    untonemappedColourJMh = tstack([untonemappedJ, M, h])

    untonemappedColourJMh[..., 1] = chromaCompressionInverse(
        JMh, untonemappedJ, params
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
