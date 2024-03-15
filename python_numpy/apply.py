import argparse
import time
import warnings

import numpy as np
import colour

from drt_init import drt_params
from drt import drt_forward


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Apply the display rendering on input ACES2065-1 image",
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
    parser.add_argument("input", action="store", type=str, help="input image path")
    parser.add_argument("output", action="store", type=str, help="output image path")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # TODO: Try to clean up warnings?
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

        inRGB = colour.read_image(args.input)[..., :3]

        start = time.time()
        outRGB = drt_forward(inRGB, params)
        end = time.time()
        print("Apply in", end - start)

        colour.write_image(outRGB, args.output, bit_depth="float32")
