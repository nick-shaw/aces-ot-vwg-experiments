import colour
from colour.models import RGB_COLOURSPACE_BT709 as BT709
from colour.algebra import spow, vector_dot
import numpy as np
import cusp_path
from cusp_path import Hellwig2022_to_XYZ, CAM_Specification_Hellwig2022, find_boundary, find_threshold
import matplotlib.pyplot as plt
import os

if not os.path.exists('./png'):
    os.makedirs('./png')

J_range = np.linspace(0, 100, 256)
XYZ_w = colour.xy_to_XYZ(BT709.whitepoint) * 100
L_A = 100.0
Y_b = 20.0
surround = colour.VIEWING_CONDITIONS_HELLWIG2022["Dim"]

for h in range(360):
    fig, ax = plt.subplots()
    M_bound = find_boundary(h, 10)
    ax.plot(M_bound, J_range, label='h={}'.format(h))
    M = M_bound.max()
    J = 100.0 * M_bound.argmax() / 255.0
    JMh = CAM_Specification_Hellwig2022(J=J, M=M, h=h)
    XYZ = Hellwig2022_to_XYZ(JMh, XYZ_w, L_A, Y_b, surround, discount_illuminant=True)
    RGB = vector_dot(BT709.matrix_XYZ_to_RGB, XYZ) / 100
    RGB = np.clip(RGB**(1/2.2), 0, 1)
    RGB_tuple = (RGB[0], RGB[1], RGB[2])
    plt.scatter(M, J, color=RGB_tuple)
    M_x = np.linspace(0, M, 256)
    J_y = J * (M_x / M)**1.15
    plt.plot(M_x, J_y)
    plt.plot([0, M], [100, J])
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('Hellwig M')
    plt.ylabel('Hellwig J')
    plt.title('Rec.709 Gamut Boundary')
    plt.legend()
    plt.savefig('./png/cusp_{:0>3}.png'.format(h))
    plt.close()
