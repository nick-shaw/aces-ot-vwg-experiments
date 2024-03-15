import argparse
import time
from unicodedata import decimal
import warnings

import numpy as np
import colour
from colour.algebra import sdiv, sdiv_mode, spow, vector_dot

from drt_init import drt_params
from drt import drt_forward, drt_inverse


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Apply the (inverse + forward) display rendering roundtrip on image.\nDefault to Colour Checker 24 values.",
    )
    parser.add_argument(
        "target",
        action="store",
        type=int,
        help="""Target for rendering:
    1: SDR sRGB 100 nits
    2: SDR BT.1886 100 nits
    3: HDR Rec2020 PQ 1000 nits""",
    )
    parser.add_argument("--img", action="store", help="input image path")
    parser.add_argument(
        "--rgb", action="store", type=float, nargs="+", help="RGB triplet"
    )
    parser.add_argument("--cube", type=int, help="RGB grid size")
    return parser.parse_args()


def hald_clut_generate(size):
    """ Generate a 3D LUT image. """

    def feval(r, g, b):
        return np.array([r, g, b])

    ax = np.linspace(0, 1, size, endpoint=True)
    x, y, z = np.meshgrid(ax, ax, ax)
    result = feval(x, y, z)
    result = np.swapaxes(result, 0, 3)

    width = size ** 2
    height = size

    result = result.reshape((height, width, 3))
    return np.ascontiguousarray(result)


if __name__ == "__main__":

    args = parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"divide by zero encountered in scalar divide")
        warnings.filterwarnings("ignore", r"divide by zero encountered in divide")
        warnings.filterwarnings("ignore", r"invalid value encountered in divide")
        warnings.filterwarnings("ignore", r"overflow encountered in power")
        warnings.filterwarnings("ignore", r"invalid value encountered in power")
        warnings.filterwarnings("ignore", r"invalid value encountered in subtract")
        warnings.filterwarnings(
            "ignore",
            r'"OpenImageIO" related API features are not available, switching to "Imageio"!',
        )

        start = time.time()
        params = drt_params(args.target)
        end = time.time()
        print("Init params in", end - start)

        if args.img:
            inRGB = colour.read_image(args.img)[..., :3]
        elif args.rgb:
            inRGB = args.rgb
        elif args.cube:
            if args.target == 3:
                scale = colour.models.eotf_inverse_BT2100_PQ(1000)
                matrix = colour.matrix_RGB_to_RGB("Display P3", "ITU-R BT.2020")
            else:
                scale = 1.0
                matrix = np.identity(3)

            inRGB = hald_clut_generate(args.cube) * scale
            inRGB = vector_dot(matrix, inRGB)
            inRGB = np.clip(inRGB, 0, scale)
        else:
            # Predefined set of test values
            # ColorChecker 24 values as per SMPTE 2065-1
            CC24 = np.array([
                [0.11877, 0.08709, 0.05895],
                [0.40002, 0.31916, 0.23736],
                [0.18476, 0.20398, 0.31311],
                [0.10901, 0.13511, 0.06493],
                [0.26684, 0.24604, 0.40932],
                [0.32283, 0.46208, 0.40606],
                [0.38605, 0.22743, 0.05777],
                [0.13822, 0.13037, 0.33703],
                [0.30202, 0.13752, 0.12758],
                [0.09310, 0.06347, 0.13525],
                [0.34876, 0.43654, 0.10613],
                [0.48655, 0.36685, 0.08061],
                [0.08732, 0.07443, 0.27274],
                [0.15366, 0.25692, 0.09071],
                [0.21742, 0.07070, 0.05130],
                [0.58919, 0.53943, 0.09157],
                [0.30904, 0.14818, 0.27426],
                [0.14901, 0.23378, 0.35939]
            ])
            inRGB = drt_forward(CC24, params)


        # Round trip test
        RGB = drt_inverse(inRGB, params)
        outRGB = drt_forward(RGB, params)

        diffRGB = np.abs(inRGB - outRGB)
        print(
            f"Roundtrip stats:\n\tMin: {np.min(diffRGB)}\n\tMean: {np.mean(diffRGB)}\n\tMed: {np.median(diffRGB)}\n\tMax: {np.max(diffRGB)}"
        )
