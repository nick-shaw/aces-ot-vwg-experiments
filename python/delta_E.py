import sys
import colour

from colour import RGB_to_ICtCp, delta_E, read_image, matrix_RGB_to_RGB
from colour.algebra import spow
from colour.utilities import tsplit, tstack

import numpy as np

import matplotlib.pyplot as plt

def strip_alpha(rgba):
    if rgba.shape[-1] == 3:
        return rgba
    else:
        r, g, b, a = tsplit(rgba)
        return tstack((r, g, b))

def main():
    if len(sys.argv) == 1:
        print('\n Usage')
        print('  python3 {} <image1> <image2> <format (optional)>'.format(sys.argv[0].split('/')[-1]))
        print()
        print('  image1: path to first image for comparison')
        print('  image2: path to second image for comparison')
        print()
        print('  format: the encoding of the images from the following options:')
        print('    BT.1886, sRGB, P3-D65, PQ (default)\n')
        exit(1)
    elif len(sys.argv) < 3:
        print('\nError: two image paths required\n')
        exit(1)

    img1 = strip_alpha(read_image(sys.argv[1]))
    img2 = strip_alpha(read_image(sys.argv[2]))

    Rec709 = colour.models.RGB_COLOURSPACE_BT709
    Rec2020 = colour.models.RGB_COLOURSPACE_BT2020
    P3D65 = colour.models.RGB_COLOURSPACE_P3_D65

    matrix_Rec709_to_Rec2020 = matrix_RGB_to_RGB(Rec709, Rec2020)
    matrix_P3D65_to_Rec2020 = matrix_RGB_to_RGB(P3D65, Rec2020)

    if len(sys.argv) > 3:
        format = sys.argv[3].upper()
        if format == 'BT.1886':
            img1 = spow(img1, 2.4)
            img2 = spow(img2, 2.4)
        elif format == 'SRGB':
            img1 = colour.eotf(img1, 'sRGB')
            img2 = colour.eotf(img2, 'sRGB')
        elif format == 'P3-D65':
            img1 = spow(img1, 2.6)
            img2 = spow(img2, 2.6)
        elif format == 'PQ':
            img1 = colour.eotf(img1, 'ST 2084')
            img2 = colour.eotf(img2, 'ST 2084')
        else:
            print('\nError: Invalid format')
    else:
        format = 'PQ'
        img1 = colour.eotf(img1, 'ST 2084')
        img2 = colour.eotf(img2, 'ST 2084')

    if format in ['BT.1886', 'SRGB']:
        img1 = colour.algebra.vecmul(matrix_Rec709_to_Rec2020, img1)
        img2 = colour.algebra.vecmul(matrix_Rec709_to_Rec2020, img2)
        img1 *= 100
        img2 *= 100
    elif format == 'P3-D65':
        img1 = colour.algebra.vecmul(matrix_P3D65_to_Rec2020, img1)
        img2 = colour.algebra.vecmul(matrix_P3D65_to_Rec2020, img2)
        img1 *= 48
        img2 *= 48

    img1_ICtCp = RGB_to_ICtCp(img1, method='ITU-R BT.2100-2 PQ')
    img2_ICtCp = RGB_to_ICtCp(img2, method='ITU-R BT.2100-2 PQ')

    delta = delta_E(img1_ICtCp, img2_ICtCp, method='ITP')

    maxDelta = np.max(delta)
    print('\nMax delta E ITP:', maxDelta, '\n')
    colour.write_image(delta, 'heatmap.tif', bit_depth='float32')

    plt.imshow(delta / maxDelta, cmap='hot')
    plt.show()

if __name__ == '__main__':
    main()
